import random
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n = len(moduli)
    
    # Precompute compatibility: two moduli are "compatible" if gcd > 1
    # (necessary condition for disjoint residue classes)
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(moduli[i], moduli[j]) > 1:
                compat[i][j] = True
                compat[j][i] = True
    
    # Build adjacency lists for faster clique search
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and compat[i][j]:
                adj[i].add(j)
    
    # Maximum clique via branch and bound
    def max_clique():
        best_clique = []
        
        def expand(clique, candidates):
            nonlocal best_clique
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            
            if not candidates:
                return
            
            # Upper bound: can't do better than current clique + all candidates
            if len(clique) + len(candidates) <= len(best_clique):
                return
            
            # Try adding each candidate
            for idx in range(len(candidates) - 1, -1, -1):
                node = candidates[idx]
                
                # Prune: if adding this node can't beat best, skip
                if len(clique) + 1 + (len(candidates) - idx - 1) <= len(best_clique):
                    continue
                
                new_clique = clique + [node]
                # New candidates must be compatible with node
                new_candidates = [c for c in candidates[:idx] if node in adj[c]]
                
                expand(new_clique, new_candidates)
        
        expand([], list(range(n)))
        return best_clique
    
    # Find maximum clique
    cliques_found = []
    
    # Try multiple random orderings for diversity
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Randomly permute and run clique search
        perm = list(range(n))
        random.shuffle(perm)
        
        # Use a simpler heuristic: greedy coloring bound for branch and bound
        def max_clique_with_order(order):
            best_clique = []
            
            def expand(clique, candidates):
                nonlocal best_clique
                if len(clique) > len(best_clique):
                    best_clique = clique[:]
                
                if not candidates:
                    return
                
                # Simple bound
                if len(clique) + len(candidates) <= len(best_clique):
                    return
                
                for idx in range(len(candidates) - 1, -1, -1):
                    node = candidates[idx]
                    if len(clique) + 1 + (len(candidates) - idx - 1) <= len(best_clique):
                        continue
                    
                    new_clique = clique + [node]
                    new_candidates = [c for c in candidates[:idx] if node in adj[c]]
                    expand(new_clique, new_candidates)
            
            expand([], order)
            return best_clique
        
        clique = max_clique_with_order(perm)
        if len(clique) > 0:
            cliques_found.append(clique)
    
    if not cliques_found:
        # Fallback: just pick compatible moduli greedily
        selected = []
        for i in range(n):
            if all(i in adj[j] for j in selected):
                selected.append(i)
        cliques_found = [selected]
    
    # Sort cliques by size descending
    cliques_found.sort(key=len, reverse=True)
    
    # For each clique (largest first), check if valid residues exist
    def find_residues_for_clique(clique_indices):
        """Try to assign residues to moduli in the clique such that all pairs are disjoint."""
        if not clique_indices:
            return None
        
        moduli_in_clique = [moduli[i] for i in clique_indices]
        n_mod = len(moduli_in_clique)
        
        # Backtracking search for residues
        residues = [None] * n_mod
        
        def backtrack(pos):
            if pos == n_mod:
                return True
            
            m = moduli_in_clique[pos]
            for r in range(m):
                # Check against all previously assigned
                valid = True
                for prev_pos in range(pos):
                    prev_m = moduli_in_clique[prev_pos]
                    prev_r = residues[prev_pos]
                    d = gcd(m, prev_m)
                    if r % d == prev_r % d:
                        valid = False
                        break
                if valid:
                    residues[pos] = r
                    if backtrack(pos + 1):
                        return True
                    residues[pos] = None
            
            return False
        
        if backtrack(0):
            return [(residues[i], moduli_in_clique[i]) for i in range(n_mod)]
        return None
    
    best_result = []
    
    # Try largest cliques first
    for clique in cliques_found[:20]:  # Try top 20 cliques
        if time.time() > deadline:
            break
        
        result = find_residues_for_clique(clique)
        if result is not None:
            if len(result) > len(best_result):
                best_result = result
    
    # If no clique worked, try greedy fallback
    if not best_result:
        current_set = []
        for i in range(n):
            m = moduli[i]
            # Check gcd condition
            ok_gcd = True
            for r_k, m_k in current_set:
                if gcd(m, m_k) == 1:
                    ok_gcd = False
                    break
            if not ok_gcd:
                continue
            
            # Find valid residue
            r_found = None
            for r in range(m):
                valid = True
                for r_k, m_k in current_set:
                    d = gcd(m, m_k)
                    if r % d == r_k % d:
                        valid = False
                        break
                if valid:
                    r_found = r
                    break
            
            if r_found is not None:
                current_set.append((r_found, m))
        
        best_result = current_set
    
    return [[r, m] for r, m in best_result]
