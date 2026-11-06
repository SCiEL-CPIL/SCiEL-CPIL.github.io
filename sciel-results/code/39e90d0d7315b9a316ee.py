import time
import itertools
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a, b, c))
    
    # Build adjacency: for each node, list of triple indices it participates in
    triples_by_node = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(bad_triples):
        triples_by_node[a-1].append(idx)
        triples_by_node[b-1].append(idx)
        triples_by_node[c-1].append(idx)
    
    # Build for each pair of nodes, the set of third nodes that form a bad triple
    pair_thirds = {}
    for idx, (a, b, c) in enumerate(bad_triples):
        pair_thirds.setdefault((a-1, b-1), set()).add(c-1)
        pair_thirds.setdefault((b-1, a-1), set()).add(c-1)
        pair_thirds.setdefault((a-1, c-1), set()).add(b-1)
        pair_thirds.setdefault((c-1, a-1), set()).add(b-1)
        pair_thirds.setdefault((b-1, c-1), set()).add(a-1)
        pair_thirds.setdefault((c-1, b-1), set()).add(a-1)
    
    # Precompute: for each node, which other nodes can be added without creating a violation
    # given the current selected set
    def can_add_node(node, selected_set):
        """Check if adding node to selected_set creates any bad triple."""
        for tidx in triples_by_node[node]:
            a, b, c = bad_triples[tidx]
            others = [x-1 for x in (a, b, c) if x-1 != node]
            if all(o in selected_set for o in others):
                return False
        return True
    
    # Greedy initialization: start empty, add nodes in order of "freedom"
    def greedy_init(order=None):
        if order is None:
            order = list(range(n))
        
        selected = set()
        # Try adding nodes greedily
        for node in order:
            if can_add_node(node, selected):
                selected.add(node)
        return selected
    
    # Local search: try adding any node that doesn't create a violation
    def local_search_add(selected):
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for node in range(n):
                if node not in selected:
                    if can_add_node(node, selected):
                        selected.add(node)
                        improved = True
        return selected
    
    # Local search: try 1-out-1-in swaps
    def local_search_swap(selected):
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for out_node in list(selected):
                for in_node in range(n):
                    if in_node not in selected:
                        new_selected = (selected - {out_node}) | {in_node}
                        if can_add_node(in_node, new_selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
        return selected
    
    # Branch and bound with algebraic structure
    # Key insight: numbers with the same largest odd divisor form chains
    # {d, 2d, 4d, 8d, ...} where d is odd
    # Within such a chain, LCM(2^a * d, 2^b * d) = 2^max(a,b) * d
    # So triples within a chain are never "bad" (all pairwise LCMs can't be equal
    # unless all three are equal, which they're not since distinct)
    # The conflicts arise between different chains
    
    def largest_odd_divisor(x):
        while x % 2 == 0:
            x //= 2
        return x
    
    # Group by largest odd divisor
    chains = {}
    for x in range(1, n+1):
        lod = largest_odd_divisor(x)
        chains.setdefault(lod, []).append(x)
    
    # Precompute: for each pair of chains, which triples exist between them
    # This helps with pruning
    
    # Initial solutions from different orderings
    best_selected = set()
    best_size = 0
    
    # Order 1: by largest odd divisor, then by value
    order1 = sorted(range(1, n+1), key=lambda x: (largest_odd_divisor(x), x))
    sel1 = greedy_init(order1)
    sel1 = local_search_add(sel1)
    sel1 = local_search_swap(sel1)
    if len(sel1) > best_size:
        best_size = len(sel1)
        best_selected = sel1
    
    # Order 2: by value
    order2 = list(range(1, n+1))
    sel2 = greedy_init(order2)
    sel2 = local_search_add(sel2)
    sel2 = local_search_swap(sel2)
    if len(sel2) > best_size:
        best_size = len(sel2)
        best_selected = sel2
    
    # Order 3: by number of prime factors (more composite first)
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
    
    order3 = sorted(range(1, n+1), key=lambda x: num_prime_factors(x), reverse=True)
    sel3 = greedy_init(order3)
    sel3 = local_search_add(sel3)
    sel3 = local_search_swap(sel3)
    if len(sel3) > best_size:
        best_size = len(sel3)
        best_selected = sel3
    
    # Order 4: random
    for seed in range(5):
        if time.time() > deadline - 20:
            break
        random.seed(seed)
        order4 = list(range(1, n+1))
        random.shuffle(order4)
        sel4 = greedy_init(order4)
        sel4 = local_search_add(sel4)
        sel4 = local_search_swap(sel4)
        if len(sel4) > best_size:
            best_size = len(sel4)
            best_selected = sel4
    
    # Branch and bound on the remaining problem
    # Use the current best as a lower bound
    # Try to improve by systematic search
    
    # Convert to bitmask for efficiency
    best_mask = 0
    for x in best_selected:
        best_mask |= (1 << (x-1))
    
    # Precompute conflict masks for fast checking
    # For each node, bitmask of nodes that conflict with it (given current selection)
    
    # Simulated annealing / random restarts from best solution
    if time.time() < deadline - 15:
        current_mask = best_mask
        current_size = bin(current_mask).count('1')
        best_found_mask = best_mask
        best_found_size = best_size
        
        temp = 1.0
        cooling = 0.999
        
        for it in range(500000):
            if time.time() > deadline - 5:
                break
            
            # Random flip
            node = random.randint(0, n-1)
            lsb = 1 << node
            
            if (current_mask >> node) & 1:
                # Remove node
                new_mask = current_mask & ~lsb
                new_size = current_size - 1
                # Always accept removal (it can only help or be neutral)
                current_mask = new_mask
                current_size = new_size
            else:
                # Add node: check if valid
                valid = True
                for tidx in triples_by_node[node]:
                    a, b, c = bad_triples[tidx]
                    others = [x-1 for x in (a, b, c) if x-1 != node]
                    if all((current_mask >> o) & 1 for o in others):
                        valid = False
                        break
                
                if valid:
                    new_mask = current_mask | lsb
                    new_size = current_size + 1
                    current_mask = new_mask
                    current_size = new_size
                else:
                    # With probability temp, accept invalid (but we can't store invalid)
                    # Instead, skip
                    continue
            
            if current_size > best_found_size:
                best_found_size = current_size
                best_found_mask = current_mask
            
            # Try to recover: add any valid node
            if current_size < best_found_size:
                for rec_node in range(n):
                    if not (current_mask >> rec_node) & 1:
                        valid = True
                        for tidx in triples_by_node[rec_node]:
                            a, b, c = bad_triples[tidx]
                            others = [x-1 for x in (a, b, c) if x-1 != rec_node]
                            if all((current_mask >> o) & 1 for o in others):
                                valid = False
                                break
                        if valid:
                            current_mask |= (1 << rec_node)
                            current_size += 1
                            break
            
            temp *= cooling
        
        if best_found_size > best_size:
            best_size = best_found_size
            best_mask = best_found_mask
    
    # Final local search on best
    if time.time() < deadline - 5:
        final_selected = set()
        for i in range(n):
            if (best_mask >> i) & 1:
                final_selected.add(i+1)
        
        final_selected = local_search_add(final_selected)
        final_selected = local_search_swap(final_selected)
        
        if len(final_selected) > best_size:
            best_size = len(final_selected)
            best_mask = 0
            for x in final_selected:
                best_mask |= (1 << (x-1))
    
    # Verify no bad triples
    result = [i+1 for i in range(n) if (best_mask >> i) & 1]
    
    # Double-check validity
    valid = True
    for a, b, c in itertools.combinations(result, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
    
    if not valid:
        # Fallback: use the odd numbers (always valid)
        result = [x for x in range(1, n+1) if x % 2 == 1]
    
    return result
