import random
from math import gcd
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # Candidate moduli: 20 to 200
    moduli = list(range(20, 201))
    n = len(moduli)
    
    # Precompute gcds and weights
    weights = [1.0 / m for m in moduli]
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Initialize with the set of even numbers (multiples of 2)
    # This is a valid clique because gcd(even, even) >= 2
    initial_set = [i for i, m in enumerate(moduli) if m % 2 == 0]
    best_set = set(initial_set)
    best_score = sum(weights[i] for i in best_set)
    
    def is_valid_addition(candidate, current_set):
        for idx in current_set:
            if gcds[candidate][idx] == 1:
                return False
        return True

    def get_score(s):
        return sum(weights[i] for i in s)

    # Local Search: Hill Climbing with random restarts
    # We use a simple neighborhood: add, remove, swap
    
    def local_search(start_set, max_iterations=500):
        current_set = set(start_set)
        current_score = get_score(current_set)
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            improved = False
            # 1. Try adding a node
            candidates_to_add = [i for i in range(n) if i not in current_set]
            random.shuffle(candidates_to_add)
            for v in candidates_to_add:
                if is_valid_addition(v, current_set):
                    new_set = current_set | {v}
                    new_score = current_score + weights[v]
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        break # Restart search loop to re-evaluate all candidates
            
            # 2. Try swapping a node
            if not improved:
                candidates_to_swap_out = list(current_set)
                random.shuffle(candidates_to_swap_out)
                for u in candidates_to_swap_out:
                    # Find a node v not in set that can replace u
                    # v must be compatible with (current_set - {u})
                    candidates_to_swap_in = [i for i in range(n) if i not in current_set]
                    random.shuffle(candidates_to_swap_in)
                    for v in candidates_to_swap_in:
                        # Check if v is compatible with all nodes in current_set except u
                        compatible = True
                        for w in current_set:
                            if w == u:
                                continue
                            if gcds[v][w] == 1:
                                compatible = False
                                break
                        if compatible:
                            new_score = current_score - weights[u] + weights[v]
                            if new_score > current_score:
                                current_set.remove(u)
                                current_set.add(v)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                        
            if not improved:
                break
                
        return current_set, current_score

    # Run multiple restarts
    # Start with even numbers, then random subsets
    best_set, best_score = local_search(initial_set)
    
    # Try other prime-based starts
    for p in [3, 5, 7, 11, 13]:
        if time.time() > deadline:
            break
        start_set = [i for i, m in enumerate(moduli) if m % p == 0]
        if start_set:
            s, sc = local_search(start_set)
            if sc > best_score:
                best_score = sc
                best_set = s

    # Try random starts
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random subset of size ~50
        rand_subset = random.sample(range(n), 50)
        # Filter to ensure it's a clique (greedy)
        clique = []
        for i in rand_subset:
            valid = True
            for j in clique:
                if gcds[i][j] == 1:
                    valid = False
                    break
            if valid:
                clique.append(i)
        
        if clique:
            s, sc = local_search(clique)
            if sc > best_score:
                best_score = sc
                best_set = s

    # Convert indices back to [m, r] format
    # We need to assign residues r such that they are pairwise disjoint.
    # Since we ensured gcd(m_i, m_j) > 1, we can always find such residues.
    # A simple assignment: r_i = 0 for all i?
    # Check: r_1 = 0, r_2 = 0.
    # Condition: 0 != 0 mod g? No, 0 == 0 mod g.
    # So r_i = 0 for all i is INVALID if gcd > 1?
    # Wait. r_1 != r_2 mod g.
    # If r_1 = 0 and r_2 = 0, then 0 == 0 mod g. This is a conflict.
    # We need r_1 != r_2 mod g.
    # We can assign r_i = i % m_i?
    # Or simply, since the graph is a clique (all pairs share a factor),
    # we can use a greedy assignment for residues.
    
    # Residue Assignment:
    # Sort moduli by size (larger first might be easier? or smaller first?)
    # Let's just assign r=0 and r=1 alternating?
    # Actually, we can always assign residues if the moduli are pairwise non-coprime.
    # Proof: We can view this as a list coloring problem.
    # But a simpler heuristic:
    # Assign r_i = 0. If conflict with previous j, change r_i to 1, 2, etc.
    # Since gcd(m_i, m_j) = g_ij, we need r_i != r_j mod g_ij.
    # There are g_ij - 1 forbidden values for r_i modulo g_ij.
    # Since m_i >= 20, and g_ij <= m_i/2 (for i != j), there are plenty of choices.
    
    selected_indices = sorted(list(best_set))
    classes = []
    assigned_residues = {}
    
    for idx in selected_indices:
        m = moduli[idx]
        r = 0
        # Check against all previously assigned
        while True:
            valid = True
            for prev_idx in assigned_residues:
                prev_m = moduli[prev_idx]
                prev_r = assigned_residues[prev_idx]
                g = gcds[idx][prev_idx]
                if (r - prev_r) % g == 0:
                    valid = False
                    break
            if valid:
                break
            r += 1
            if r >= m:
                # Should not happen, but fallback
                r = 0
                break
        
        assigned_residues[idx] = r
        classes.append([m, r])
        
    return classes

