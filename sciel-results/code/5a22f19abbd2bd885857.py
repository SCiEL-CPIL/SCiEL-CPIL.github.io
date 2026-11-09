"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix for speed
    gcd_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g
            
    # Precompute coprime adjacency for fast checking
    # coprime_adj[i] is a bitmask of indices j such that gcd(all_moduli[i], all_moduli[j]) == 1
    coprime_adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and gcd_matrix[i][j] == 1:
                mask |= (1 << j)
        coprime_adj[i] = mask

    def is_valid_set(selected_indices):
        """Check if the set of indices forms a pairwise non-coprime set."""
        # Use bitsets for fast intersection check
        # A set is valid if for every i in set, (coprime_adj[i] & set_mask) == 0
        set_mask = 0
        for idx in selected_indices:
            set_mask |= (1 << idx)
        
        for idx in selected_indices:
            if coprime_adj[idx] & set_mask:
                return False
        return True

    def get_size(selected_indices):
        return len(selected_indices)

    # Initial Solution: All Even Numbers
    # Indices of even numbers
    initial_set = [i for i in range(n) if all_moduli[i] % 2 == 0]
    
    # Verify initial set is valid (it should be, as all evens share factor 2)
    if not is_valid_set(initial_set):
        # Fallback to single element if logic fails (should not happen)
        initial_set = [0] 

    best_set = initial_set[:]
    best_score = len(best_set)
    
    # Local Search / Hill Climbing with Random Restarts
    # We try to improve the set by adding, removing, or swapping elements.
    # Since the constraint is hard (pairwise non-coprime), we can't just accept invalid states easily 
    # without a complex penalty function. 
    # Instead, we can perform "move" operations that preserve validity or use a repair mechanism.
    
    # A simpler effective strategy for this specific problem:
    # Start with all evens. 
    # Try to replace an even number with an odd number if the odd number is compatible with all remaining evens and other odds.
    # Or, more generally, use a randomized greedy construction with restarts.
    
    def random_greedy():
        """Construct a set using randomized greedy."""
        # Shuffle moduli
        candidates = list(range(n))
        random.shuffle(candidates)
        
        selected = []
        selected_mask = 0
        
        for idx in candidates:
            # Check if idx is coprime to any already selected
            # If coprime_adj[idx] & selected_mask == 0, it is compatible
            if coprime_adj[idx] & selected_mask == 0:
                selected.append(idx)
                selected_mask |= (1 << idx)
                
        return selected

    def local_improve(current_set):
        """Try to improve a valid set by swapping or adding."""
        current_set = current_set[:]
        current_mask = 0
        for idx in current_set:
            current_mask |= (1 << idx)
            
        improved = True
        while improved:
            improved = False
            
            # Try adding a new element
            # Find all indices not in current_set
            # If adding idx keeps it valid, add it.
            # Since we start with a maximal set (greedy or evens), we might not be able to add directly.
            # So we try swapping: remove i, add j.
            
            # Optimization: Precompute potential swaps?
            # For now, simple random swaps.
            
            # Try to add an element that is compatible with all current elements
            # This is only possible if current_set is not maximal.
            # Let's check if we can add any.
            for idx in range(n):
                if (current_mask >> idx) & 1:
                    continue
                if coprime_adj[idx] & current_mask == 0:
                    current_set.append(idx)
                    current_mask |= (1 << idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove i, add j
            # We want to increase size or maintain size but allow future additions?
            # Since we maximize size, swaps that maintain size are only useful if they enable additions.
            # However, simple hill climbing on size only accepts improvements.
            # Let's try to find a swap that results in a set that allows an addition?
            # That's complex. 
            # Alternative: Just try random swaps and see if the new set allows an addition immediately after.
            
            # For efficiency, let's just try to find any valid set larger than current via random restarts.
            # The "All Evens" set is size 63. 
            # Can we do better?
            # The maximum independent set in the coprime graph of [2,128] is likely around 63-70.
            # Let's rely on Randomized Greedy with many restarts to find a large set.
            
            # If we can't add, we stop local improvement for this path.
            break
            
        return current_set

    # Strategy: Run many randomized greedy constructions and keep the best.
    # Also try to "repair" or "extend" the best set found.
    
    best_result = best_set[:]
    
    num_restarts = 2000
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Randomized Greedy
        # To make it more effective, we can bias the shuffle or use weighted random selection.
        # Simple shuffle is often good enough for this size.
        
        candidates = list(range(n))
        # Bias towards numbers with many non-coprime neighbors?
        # Degrees in the non-coprime graph:
        # deg[i] = n - 1 - popcount(coprime_adj[i])
        
        # Let's just shuffle for now.
        random.shuffle(candidates)
        
        selected = []
        selected_mask = 0
        
        for idx in candidates:
            if coprime_adj[idx] & selected_mask == 0:
                selected.append(idx)
                selected_mask |= (1 << idx)
                
        if len(selected) > len(best_result):
            best_result = selected[:]
            
        # Try to improve best_result locally?
        # We already have a local improvement that tries to add.
        # Let's apply it to the current best_result periodically.
        
        if i % 10 == 0:
            improved_best = local_improve(best_result)
            if len(improved_best) > len(best_result):
                best_result = improved_best

    # Final local improvement on the best found
    final_set = local_improve(best_result)
    
    # Assign Residues
    # We have a list of indices. Convert to moduli.
    selected_moduli_indices = final_set
    selected_moduli = [all_moduli[i] for i in selected_moduli_indices]
    
    # Greedy Residue Assignment
    # For each modulus, pick a residue r such that for all previously assigned (r_k, m_k),
    # r % gcd(m, m_k) != r_k % gcd(m, m_k)
    
    result = []
    # Sort by modulus? Or just process in order.
    # To make it easier, maybe process smaller moduli first? Or larger?
    # Let's try processing in the order they were selected.
    
    for m in selected_moduli:
        found = False
        # Try residues 0 to m-1
        for r in range(m):
            compatible = True
            for r_prev, m_prev in result:
                g = gcd(m, m_prev)
                if r % g == r_prev % g:
                    compatible = False
                    break
            if compatible:
                result.append((r, m))
                found = True
                break
        
        if not found:
            # If greedy fails, we might need backtracking.
            # But for pairwise non-coprime sets, greedy usually works because constraints are loose.
            # If it fails, we might have to drop this modulus and try to re-add others?
            # For now, if it fails, we just skip it (reducing the count).
            # This is a risk. Let's implement a simple backtracking for residue assignment if greedy fails.
            pass

    # Let's implement a robust residue assignment with backtracking just in case.
    def assign_residues(moduli_list):
        """
        Assign residues to moduli_list such that pairwise disjointness is maintained.
        Returns list of (residue, modulus) or None if impossible.
        """
        n_m = len(moduli_list)
        residues = [None] * n_m
        
        def backtrack(i):
            if i == n_m:
                return True
            
            m = moduli_list[i]
            for r in range(m):
                valid = True
                for j in range(i):
                    if residues[j] is None:
                        continue
                    m_j = moduli_list[j]
                    r_j = residues[j]
                    g = gcd(m, m_j)
                    if r % g == r_j % g:
                        valid = False
                        break
                
                if valid:
                    residues[i] = r
                    if backtrack(i + 1):
                        return True
                    residues[i] = None
                    
            return False
        
        if backtrack(0):
            return [(residues[i], moduli_list[i]) for i in range(n_m)]
        else:
            return None

    assigned = assign_residues(selected_moduli)
    
    if assigned is None:
        # Fallback: Try to find a valid subset of moduli that allows residue assignment.
        # This is unlikely if the moduli are pairwise non-coprime, but let's be safe.
        # We can just return the first few that work.
        result = []
        for m in selected_moduli:
            for r in range(m):
                compatible = True
                for r_prev, m_prev in result:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    result.append((r, m))
                    break
    else:
        result = assigned

    # Convert to list of lists
    return [[r, m] for r, m in result]

# EVOLVE-BLOCK-END
