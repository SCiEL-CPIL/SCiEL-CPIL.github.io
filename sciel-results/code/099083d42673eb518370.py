import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Tighter safety margin
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lcm_table[i][j] = i // gcd(i, j) * j
    
    # Build conflict graph: edge between u,v if they appear together in some bad triple
    # A set is valid iff it contains no three elements that form a bad triple
    # Equivalently: for every bad triple (a,b,c), we cannot have all three in the set
    # This is a 3-uniform hypergraph independent set problem
    
    # conflict_pairs[x]: set of (u, v) such that {u, v, x} is a bad triple
    conflict_pairs = [set() for _ in range(n + 1)]
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_triples.append((a, b, c))
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    # Build pairwise conflict graph: u and v conflict if they appear in a bad triple together
    # But we need the 3-uniform structure. Let's use a different approach.
    # 
    # For branch-and-bound, we need to check if adding element x to current set S
    # creates a bad triple. This happens if there exist u,v in S such that (u,v,x) is bad.
    # 
    # Let's precompute for each x, the set of pairs (u,v) that conflict with x.
    # Then when checking if x can be added to S, we check if any such pair is fully in S.
    
    # For efficiency, let's represent the "forbidden pairs for x" as a bitset of pairs
    # Actually, let's use a different representation:
    # For each x, store list of (u,v) pairs. When checking can_add(x, S),
    # we check if both u and v are in S for any (u,v) in conflict_pairs[x].
    
    # Precompute degrees
    degrees = [len(conflict_pairs[i]) for i in range(n + 1)]
    
    # Precompute for each x, the conflict pairs as a list of (bit_u, bit_v) for faster checking
    conflict_bits = [None] * (n + 1)
    for x in range(1, n + 1):
        conflict_bits[x] = [(1 << (u - 1), 1 << (v - 1)) for u, v in conflict_pairs[x]]
    
    def can_add(x, mask):
        """Check if element x can be added to the set represented by mask."""
        for bu, bv in conflict_bits[x]:
            if (mask & bu) and (mask & bv):
                return False
        return True
    
    def popcount(mask):
        # Fast popcount using int.bit_count() (Python 3.10+)
        return mask.bit_count()
    
    def greedy(order, mask=0):
        """Greedy construction given an order of elements."""
        selected = mask
        for x in order:
            if not (selected >> (x - 1)) & 1:
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected
    
    def local_search(mask):
        """
        Fast local search:
        1. Greedy add any valid element.
        2. If stuck, try removing one or two random elements and greedily filling again.
        """
        improved = True
        best_local = mask
        best_local_size = popcount(mask)
        
        while improved and time.time() < deadline - 3:
            improved = False
            
            # 1. Greedy add pass
            for x in range(1, n + 1):
                if not (mask >> (x - 1)) & 1:
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
            
            sz = popcount(mask)
            if sz > best_local_size:
                best_local = mask
                best_local_size = sz
            
            if not improved:
                # 2. Perturbation: Remove one or two, fill greedily
                s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                if not s_list:
                    break
                
                # Try random removals to escape local optima
                for _ in range(50):
                    if time.time() > deadline - 3:
                        break
                    if random.random() < 0.5:
                        # Remove one
                        x = random.choice(s_list)
                        mask_no_x = mask & ~(1 << (x - 1))
                    else:
                        # Remove two
                        if len(s_list) < 2:
                            continue
                        x1, x2 = random.sample(s_list, 2)
                        mask_no_x = mask & ~(1 << (x1 - 1)) & ~(1 << (x2 - 1))
                    
                    new_mask = mask_no_x
                    # Greedy fill in random order for diversity
                    fill_order = list(range(1, n + 1))
                    random.shuffle(fill_order)
                    for y in fill_order:
                        if not (new_mask >> (y - 1)) & 1:
                            if can_add(y, new_mask):
                                new_mask |= (1 << (y - 1))
                    
                    new_sz = popcount(new_mask)
                    if new_sz > best_local_size:
                        best_local = new_mask
                        best_local_size = new_sz
                        mask = new_mask
                        improved = True
                        s_list = [i for i in range(1, n + 1) if (mask >> (i - 1)) & 1]
                        break
                        
        return best_local
    
    best_mask = 0
    best_size = 0
    
    # Phase 1: Multi-start Greedy + Fast Local Search to get a good initial solution
    num_restarts = 0
    while time.time() < deadline - 5:  # Use more time for search
        num_restarts += 1
        
        # Generate diverse orders
        r = random.random()
        if r < 0.3:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif r < 0.5:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x])
        elif r < 0.7:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
        elif r < 0.85:
            # Weighted random by inverse degree
            weights = [1.0 / max(1, degrees[x]) for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            random.shuffle(order)
            # Sort by random key weighted by inverse degree
            order = sorted(order, key=lambda x: random.random() * degrees[x])
        else:
            # Prime-priority order
            def is_prime(k):
                if k < 2:
                    return False
                for d in range(2, int(k**0.5) + 1):
                    if k % d == 0:
                        return False
                return True
            order = sorted(range(1, n + 1), key=lambda x: (0 if is_prime(x) else 1, x))
            
        mask = greedy(order)
        mask = local_search(mask)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Phase 2: Branch and Bound to try to improve
    # We'll work with the complement: elements NOT in the best solution
    # Try to find a larger independent set
    
    # Build adjacency for the "conflict" relation in a way suitable for B&B
    # Two elements u, v are "compatible" if they can coexist in a valid set
    # But compatibility depends on the rest of the set...
    # 
    # Actually, let's use a different approach for B&B:
    # We search for the maximum valid subset.
    # State: (candidate_set, current_set)
    # At each step, we either include or exclude the next candidate.
    # Upper bound: current_size + len(candidate_set)
    # We prune if upper_bound <= best_size
    
    # For better bounds, we can use a greedy coloring on the "conflict graph"
    # where u and v are adjacent if they cannot both be in a valid set WITH THE CURRENT PARTIAL SOLUTION
    # This is complex. Let's use a simpler bound.
    
    # Actually, let's try a different B&B formulation:
    # We maintain a partial assignment: included, excluded, undecided
    # For each undecided element, we check if it can be included given the included set
    # Upper bound: |included| + max_valid_subset_of(undecided, given included)
    # For the upper bound, we can use |undecided| as a crude bound, or compute a better one
    
    # Let's implement a simple B&B with the crude bound first
    def branch_and_bound(undecided_mask, included_mask, current_size):
        """
        Branch and bound to find maximum valid set.
        undecided_mask: bits for elements not yet decided
        included_mask: bits for elements already included
        current_size: popcount of included_mask
        Returns (best_size_found, best_mask)
        """
        nonlocal best_size, best_mask
        
        if time.time() > deadline - 2:
            return
        
        if undecided_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = included_mask
            return
        
        # Upper bound: current_size + popcount(undecided_mask)
        if current_size + popcount(undecided_mask) <= best_size:
            return
        
        # Choose next element to branch on
        # Pick the element with highest degree in the undecided set for better pruning
        # For simplicity, pick the lowest bit
        x = (undecided_mask & -undecided_mask).bit_length()
        # x is the bit position (1-indexed), so element is x
        # Actually, bit_length gives the position of the highest set bit
        # Let's use a different method to get the lowest set bit
        lowest_bit = undecided_mask & -undecided_mask
        x = lowest_bit.bit_length()  # This gives 1-indexed position
        
        # Branch 1: Exclude x
        new_undecided = undecided_mask & ~lowest_bit
        branch_and_bound(new_undecided, included_mask, current_size)
        
        # Branch 2: Include x (if possible)
        if can_add(x, included_mask):
            new_included = included_mask | lowest_bit
            branch_and_bound(new_undecided, new_included, current_size + 1)
    
    # Start B&B from scratch with all elements undecided
    # But this might be too slow. Let's use the best solution as a starting point
    # and try to improve by swapping elements
    
    # Alternative: Use B&B to verify if we can do better than best_size
    # Start with all elements, try to find a set of size > best_size
    
    # Let's use a smarter approach: 
    # 1. Get a good initial solution from Phase 1
    # 2. Use local search with more aggressive perturbation
    # 3. Use a limited B&B that only explores promising branches
    
    # For now, let's enhance the local search with simulated annealing-like cooling
    
    def enhanced_local_search(mask):
        """Enhanced local search with simulated annealing."""
        current_mask = mask
        current_size = popcount(mask)
        best_local = mask
        best_local_size = current_size
        
        temperature = 5.0
        cooling = 0.99
        max_iterations = 10000
        no_improve_count = 0
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            if temperature < 0.01:
                break
            
            # Random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.25:
                # Try to add a random element
                available = [i for i in range(1, n + 1) if not (current_mask >> (i - 1)) & 1]
                if not available:
                    continue
                x = random.choice(available)
                if can_add(x, current_mask):
                    new_mask = current_mask | (1 << (x - 1))
                    delta = 1
                else:
                    continue
            elif move_type < 0.45:
                # Try to remove a random element
                in_set = [i for i in range(1, n + 1) if (current_mask >> (i - 1)) & 1]
                if not in_set:
                    continue
                x = random.choice(in_set)
                new_mask = current_mask & ~(1 << (x - 1))
                delta = -1
            elif move_type < 0.75:
                # Try to swap: remove one, add one
                in_set = [i for i in range(1, n + 1) if (current_mask >> (i - 1)) & 1]
                available = [i for i in range(1, n + 1) if not (current_mask >> (i - 1)) & 1]
                if not in_set or not available:
                    continue
                x_remove = random.choice(in_set)
                x_add = random.choice(available)
                mask_without = current_mask & ~(1 << (x_remove - 1))
                if can_add(x_add, mask_without):
                    new_mask = mask_without | (1 << (x_add - 1))
                    delta = 0
                else:
                    continue
            else:
                # Try double swap: remove two, add two
                in_set = [i for i in range(1, n + 1) if (current_mask >> (i - 1)) & 1]
                available = [i for i in range(1, n + 1) if not (current_mask >> (i - 1)) & 1]
                if len(in_set) < 2 or len(available) < 2:
                    continue
                x1, x2 = random.sample(in_set, 2)
                y1, y2 = random.sample(available, 2)
                mask_without = current_mask & ~(1 << (x1 - 1)) & ~(1 << (x2 - 1))
                if can_add(y1, mask_without):
                    mask_mid = mask_without | (1 << (y1 - 1))
                    if can_add(y2, mask_mid):
                        new_mask = mask_mid | (1 << (y2 - 1))
                        delta = 0
                    else:
                        continue
                else:
                    continue
            
            new_size = popcount(new_mask)
            if new_size > best_local_size:
                best_local = new_mask
                best_local_size = new_size
                current_mask = new_mask
                current_size = new_size
                no_improve_count = 0
            else:
                # Accept with probability exp(delta / temperature)
                if delta >= 0:
                    current_mask = new_mask
                    current_size = new_size
                else:
                    prob = 2.718281828 ** (delta / temperature)
                    if random.random() < prob:
                        current_mask = new_mask
                        current_size = new_size
                no_improve_count += 1
            
            temperature *= cooling
        
        return best_local
    
    # Run enhanced local search on the best solution with more iterations
    if time.time() < deadline - 2:
        best_mask = enhanced_local_search(best_mask)
        best_size = popcount(best_mask)
    
    # Final greedy polish - multiple passes
    for _ in range(3):
        changed = False
        for x in range(1, n + 1):
            if not (best_mask >> (x - 1)) & 1:
                if can_add(x, best_mask):
                    best_mask |= (1 << (x - 1))
                    best_size += 1
                    changed = True
        if not changed:
            break
    
    # Final local search polish
    if time.time() < deadline - 1:
        best_mask = local_search(best_mask)
        best_size = popcount(best_mask)
    
    result = [i for i in range(1, n + 1) if (best_mask >> (i - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(res)
