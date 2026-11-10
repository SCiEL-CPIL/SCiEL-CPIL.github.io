import time
import random
import math
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability: for each mask, list of masks it's comparable with
    # comp_list[a] = list of b != a where a and b are comparable
    comp_list = [[] for _ in range(num_masks)]
    for a in range(num_masks):
        for b in range(a + 1, num_masks):
            if (a & b) == a or (a & b) == b:
                comp_list[a].append(b)
                comp_list[b].append(a)
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                # Check if comparable
                a, b = masks[i], masks[j]
                if (a & b) == a or (a & b) == b:
                    s += 1
        return s
    
    def solve_max_weight_closure(lambdas):
        """
        Solve max weight closure on DAG using min-cut.
        Nodes: 0..num_masks-1 represent subsets.
        Edge a->b (a subset of b) with infinite capacity forces: if a selected, b must be selected.
        But we want to select exactly m nodes. Use Lagrangian: 
          maximize sum_{i<j, comparable} 1 - lambda * |S|
        This is not directly a closure problem. Instead, use a different approach.
        
        Alternative: Model as maximum weight closure where:
        - Each node v has weight w_v = lambda (the Lagrange multiplier for cardinality)
        - For each comparable pair (a,b), we get a bonus of 1 if both are selected.
        
        This is a quadratic objective. Convert to closure by introducing auxiliary nodes.
        
        Simpler approach: Use min-cut with:
        - Source -> node v with capacity w_v (if w_v > 0)
        - Node v -> sink with capacity -w_v (if w_v < 0)
        - For each "must include together" constraint, infinite capacity edge.
        
        For our problem, the comparability bonus is pairwise. We can use the standard
        reduction: for each comparable pair (a,b), add an auxiliary node e_ab with 
        infinite capacity edges a->e_ab and b->e_ab, and e_ab->sink with capacity 1.
        Then selecting both a and b "activates" e_ab and loses 1, so we maximize by NOT
        selecting it, giving +1 bonus.
        
        Actually, let's use a simpler heuristic-based approach with the DAG structure.
        """
        # Build the DAG: edge from a to b if a is strict subset of b
        # For min-cut formulation of max weight closure:
        # - If w_v > 0: source -> v with capacity w_v
        # - If w_v < 0: v -> sink with capacity -w_v
        # - For each edge (u,v) in DAG (u must be selected if v is): u -> v with INF
        
        # But our objective is quadratic. Let's use a different strategy:
        # Greedy + local search with DAG-aware moves.
        
        # For now, return a greedy solution based on lambdas
        # Weight of selecting mask v = lambda + (number of comparable masks already selected)
        # This is approximate.
        
        # Instead, let's just do a good greedy + local search
        # The "DAG" insight: prioritize chains (totally ordered subsets)
        
        # Build chains: for each mask, the chain is all its subsets
        # A good solution should have many overlapping chains
        
        # Greedy: start with a long chain, then add sets that maximize comparability
        # Chain: 0, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
        chain = [0]
        for i in range(n):
            chain.append((1 << (i + 1)) - 1)
        
        # Add more sets greedily
        selected = list(chain[:m]) if len(chain) >= m else list(chain)
        used = set(selected)
        
        # Add remaining greedily
        for _ in range(m - len(selected)):
            best_mask = -1
            best_gain = -1
            for mask in range(num_masks):
                if mask in used:
                    continue
                gain = 0
                for sm in selected:
                    if (mask & sm) == mask or (mask & sm) == sm:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask != -1:
                selected.append(best_mask)
                used.add(best_mask)
        
        return selected
    
    # 1. Initial solution: use chain-based greedy
    # Build a good initial solution using multiple chains
    random.seed(42)
    
    # Strategy: Use several overlapping chains to maximize comparability
    # Chain 1: 0, {0}, {0,1}, ..., {0..11}
    # Chain 2: 0, {1}, {0,1}, {0,1,2}, ...
    # etc.
    
    # Build initial solution with chain structure
    current_masks = []
    used_masks = set()
    
    # Add empty and full set
    current_masks.append(0)
    used_masks.add(0)
    current_masks.append(full_mask)
    used_masks.add(full_mask)
    
    # Add sets from a primary chain
    for i in range(1, n):
        if len(current_masks) >= m:
            break
        mask = (1 << i) - 1
        if mask not in used_masks:
            current_masks.append(mask)
            used_masks.add(mask)
    
    # Add more sets greedily
    for _ in range(m - len(current_masks)):
        best_mask = -1
        best_score = -1
        for mask in range(1, full_mask):
            if mask in used_masks:
                continue
            score = 0
            for cm in current_masks:
                if (mask & cm) == mask or (mask & cm) == cm:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        else:
            mask = random.randint(0, full_mask)
            while mask in used_masks:
                mask = random.randint(0, full_mask)
            current_masks.append(mask)
            used_masks.add(mask)

    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # 2. Local search with DAG-aware moves
    # Key insight: moves that stay within or near chains are more likely to be good
    
    T = 10.0
    cooling_rate = 0.9998
    min_T = 0.01
    
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 10.0
        no_improve_steps = 0
        max_no_improve = 15000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask with DAG-aware bias
            r = random.random()
            if r < 0.3:
                # Move within the same "chain neighborhood"
                # Add or remove one bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.5:
                # Move to a nearby set in the subset lattice (Hamming distance 2)
                if old_mask == 0:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = (1 << b1) | (1 << b2)
                elif old_mask == full_mask:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
                else:
                    # Try to add or remove 2 bits
                    if random.random() < 0.5:
                        # Remove 2 bits
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        if len(bits) >= 2:
                            b1, b2 = random.sample(bits, 2)
                            new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
                        else:
                            b1 = random.choice(bits)
                            new_mask = old_mask ^ (1 << b1)
                    else:
                        # Add 2 bits
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        if len(bits) >= 2:
                            b1, b2 = random.sample(bits, 2)
                            new_mask = old_mask | (1 << b1) | (1 << b2)
                        elif len(bits) == 1:
                            new_mask = old_mask | (1 << bits[0])
                        else:
                            new_mask = old_mask
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                cm = current_masks[j]
                old_comp = 1 if ((old_mask & cm) == old_mask or (old_mask & cm) == cm) else 0
                new_comp = 1 if ((new_mask & cm) == new_mask or (new_mask & cm) == cm) else 0
                delta += new_comp - old_comp
            
            # Acceptance
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[idx] = new_mask
                    current_score += delta
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break

        restarts += 1

    # 3. Final polish: try all single-swap moves
    for _ in range(50):
        improved = False
        for i in range(m):
            if time.time() > deadline - 2:
                break
            for j in range(i + 1, m):
                if time.time() > deadline - 2:
                    break
                # Try swapping masks[i] and masks[j] with other available masks
                # Actually, just try replacing one at a time
                pass
        
        # Better polish: try replacing each element with any other mask
        for i in range(m):
            if time.time() > deadline - 2:
                break
            old_mask = best_solution[i]
            best_new_mask = old_mask
            best_delta = 0
            for new_mask in range(num_masks):
                if new_mask == old_mask:
                    continue
                if new_mask in set(best_solution):
                    continue
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    cm = best_solution[j]
                    old_comp = 1 if ((old_mask & cm) == old_mask or (old_mask & cm) == cm) else 0
                    new_comp = 1 if ((new_mask & cm) == new_mask or (new_mask & cm) == cm) else 0
                    delta += new_comp - old_comp
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_delta > 0:
                best_solution[i] = best_new_mask
                best_score += best_delta
                improved = True
        
        if not improved:
            break

    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
