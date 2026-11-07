"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute subset relation: sub[i][j] = True if i is a strict subset of j
    # sub[i][j] means i ⊂ j (i is proper subset of j)
    # Comparable if i⊂j or j⊂i
    sub = np.zeros((N, N), dtype=bool)
    for i in range(N):
        # All supersets of i
        # A superset of i has all bits of i plus possibly more
        # Enumerate supersets: take complement, iterate subsets of complement
        comp = (~i) & (N - 1)
        s = comp
        while True:
            j = i | s
            if j != i:  # strict superset
                sub[i][j] = True
            if s == 0:
                break
            s = (s - 1) & comp
    # Now sub[i][j] = True means i ⊂ j
    # Comparable matrix: comp_mat[i][j] = sub[i][j] or sub[j][i]
    comp_mat = sub | sub.T
    
    def compute_scores(family):
        """Compute score vector S where S[i] = number of j≠i with comp_mat[family[i]][family[j]]"""
        fam_arr = np.array(family)
        # For each i, count comparable with all others
        # comp_mat[fam_arr[i], fam_arr] gives boolean vector
        # Subtract 1 for self (comp_mat[x][x] is False, so no need)
        scores = np.array([comp_mat[fam_arr[i], fam_arr].sum() for i in range(m)])
        return scores
    
    def total_comparable_pairs(family):
        """Count total unordered comparable pairs"""
        fam_set = set(family)
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                if comp_mat[family[i]][family[j]]:
                    count += 1
        return count
    
    def random_chain_family(rng):
        """Generate a random family with many chain structures"""
        family = []
        used = set()
        
        # Create several chains of random subsets
        # A chain is a sequence where each is a subset of the next
        num_chains = rng.randint(5, 10)
        remaining = m
        for c in range(num_chains):
            if remaining <= 0:
                break
            chain_len = rng.randint(2, min(remaining, 8))
            # Start with a random small set
            cur = 0
            for _ in range(rng.randint(0, 3)):
                if cur != 0:
                    break
                cur = 1 << rng.randint(0, n - 1)
            
            chain = []
            for step in range(chain_len):
                if cur in used:
                    # Try to add a random bit
                    available = [b for b in range(n) if (cur >> b) & 1 == 0]
                    if available:
                        cur |= (1 << rng.choice(available))
                    else:
                        break
                if cur in used:
                    continue
                chain.append(cur)
                used.add(cur)
                # Add a random bit to grow
                available = [b for b in range(n) if (cur >> b) & 1 == 0]
                if available:
                    cur |= (1 << rng.choice(available))
                else:
                    break
            
            family.extend(chain)
            remaining -= len(chain)
        
        # Fill remaining with random sets
        while len(family) < m:
            s = rng.randint(0, N - 1)
            if s not in used:
                family.append(s)
                used.add(s)
        
        return family[:m]
    
    def local_search(family, max_time=5.0):
        """Perform local search to improve the family"""
        family = list(family)
        fam_set = set(family)
        S = compute_scores(family)
        best_family = list(family)
        best_score = total_comparable_pairs(family)
        
        elapsed = time.time() - start_time
        ls_deadline = deadline - 2.0  # leave 2s margin
        
        improved = True
        while improved and time.time() < ls_deadline:
            improved = False
            fam_set = set(family)
            
            for i in range(m):
                if time.time() > ls_deadline:
                    break
                
                old_mask = family[i]
                old_S = S[i]
                
                best_mask = -1
                best_s_m = -1
                
                # Find best replacement
                # For each candidate M not in family, compute how many existing family members are comparable to M
                # This is sum over j≠i of comp_mat[family[j]][M]
                # We can precompute for all M at once using matrix operations
                
                # Build a boolean vector of which family members are comparable to each candidate
                # comp_mat[:, M] for all M at once is just comp_mat (columns)
                # For candidate M, score = sum of comp_mat[family[j]][M] for j≠i
                # = sum of comp_mat[family[j], M] over all j - comp_mat[family[i], M]
                
                # Precompute: for each family member j, comp_mat[family[j], :] is a row
                # Sum over all j: total_comp[M] = sum_j comp_mat[family[j]][M]
                # Then for candidate M replacing i: score = total_comp[M] - comp_mat[family[i]][M]
                
                # Compute total_comp
                total_comp = np.zeros(N, dtype=np.int32)
                for j in range(m):
                    total_comp += comp_mat[family[j]]
                
                # Score for each candidate M
                scores = total_comp - comp_mat[old_mask]
                # Mask out existing family members
                for f in family:
                    scores[f] = -1
                # Also mask out old_mask itself (already in family, but ensure)
                scores[old_mask] = -1
                
                best_idx = np.argmax(scores)
                best_s_m = int(scores[best_idx])
                best_mask = int(best_idx)
                
                if best_mask >= 0 and best_s_m > old_S:
                    # Perform swap
                    for j in range(m):
                        if j == i:
                            continue
                        delta_j = 0
                        if comp_mat[best_mask][family[j]]:
                            delta_j += 1
                        if comp_mat[old_mask][family[j]]:
                            delta_j -= 1
                        S[j] += delta_j
                    
                    S[i] = best_s_m
                    
                    fam_set.remove(old_mask)
                    fam_set.add(best_mask)
                    family[i] = best_mask
                    
                    improved = True
            
            # Check if we improved
            cur_score = total_comparable_pairs(family)
            if cur_score > best_score:
                best_score = cur_score
                best_family = list(family)
        
        return best_family, best_score
    
    # Try multiple random restarts
    best_overall_family = None
    best_overall_score = 0
    
    # First, try a structured initial family
    # Use chains based on binary representations
    rng = random.Random(42)
    
    # Create initial family with chain structure
    # Use subsets of a fixed set of bits to create natural chains
    init_family = []
    used = set()
    
    # Create chains using different bit patterns
    # Chain 1: single bits {0}, {0,1}, {0,1,2}, ...
    for length in range(1, 13):
        mask = (1 << length) - 1
        if mask not in used:
            init_family.append(mask)
            used.add(mask)
    
    # Chain 2: different pattern
    mask = 0
    for b in [5, 6, 7, 8, 9, 10, 11]:
        mask |= (1 << b)
        if mask not in used:
            init_family.append(mask)
            used.add(mask)
    
    # Chain 3
    mask = 1 << 0
    for b in [2, 4, 6, 8, 10]:
        mask |= (1 << b)
        if mask not in used:
            init_family.append(mask)
            used.add(mask)
    
    # Fill remaining with random sets that might be comparable
    while len(init_family) < m:
        s = rng.randint(0, N - 1)
        if s not in used:
            init_family.append(s)
            used.add(s)
    
    init_family = init_family[:m]
    init_score = total_comparable_pairs(init_family)
    
    if init_score > best_overall_score:
        best_overall_score = init_score
        best_overall_family = list(init_family)
    
    # Local search on initial family
    if time.time() < deadline - 10:
        ls_family, ls_score = local_search(init_family)
        if ls_score > best_overall_score:
            best_overall_score = ls_score
            best_overall_family = ls_family
    
    # Try random restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(1000 + r)
        rand_family = random_chain_family(rng)
        rand_score = total_comparable_pairs(rand_family)
        
        if rand_score > best_overall_score:
            best_overall_score = rand_score
            best_overall_family = list(rand_family)
        
        # Local search
        if time.time() < deadline - 10:
            ls_family, ls_score = local_search(rand_family)
            if ls_score > best_overall_score:
                best_overall_score = ls_score
                best_overall_family = ls_family
    
    # Final local search on best family found
    if best_overall_family and time.time() < deadline - 5:
        final_family, final_score = local_search(best_overall_family)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_family = final_family
    
    # Convert to lists of integers
    result = []
    for mask in best_overall_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
