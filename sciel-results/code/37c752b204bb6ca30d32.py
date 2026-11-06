import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    # All possible triples as sorted tuples
    all_triples = [tuple(sorted(t)) for t in combinations(range(16), 3)]
    all_triples_set = set(all_triples)
    
    # Precompute: for each triple, which other triples form a K(2,2,2) with it
    # A K(2,2,2) consists of 8 transversals of {a,a'}x{b,b'}x{c,c'}
    # For triple T = (a,b,c), a conflict arises if there exist a',b',c' such that
    # all 8 triples from {a,a'}x{b,b'}x{c,c'} are present (including T)
    
    # Build conflict graph: for each triple, list of triples that would complete a K(2,2,2) with it
    # Actually, we need: adding T creates conflict iff there exist a',b',c' such that the other 7
    # transversals are already in E.
    
    # For faster lookup, use a set-based membership test
    def creates_conflict(E, T):
        """
        Returns True if adding T to E creates a K(2,2,2).
        """
        a, b, c = T
        # We need to check if there exist a', b', c' (all distinct from respective)
        # such that all 8 transversals of {a,a'}x{b,b'}x{c,c'} are in E∪{T}
        # Since T=(a,b,c) is one of them, we need the other 7 in E.
        
        # The 8 transversals are:
        # (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
        
        for a_prime in range(16):
            if a_prime == a: continue
            for b_prime in range(16):
                if b_prime == b: continue
                # Need (a', b, c) and (a, b', c) in E
                if tuple(sorted((a_prime, b, c))) not in E: continue
                if tuple(sorted((a, b_prime, c))) not in E: continue
                for c_prime in range(16):
                    if c_prime == c: continue
                    # Need remaining 5 in E:
                    if tuple(sorted((a, b, c_prime))) not in E: continue
                    if tuple(sorted((a_prime, b_prime, c))) not in E: continue
                    if tuple(sorted((a_prime, b, c_prime))) not in E: continue
                    if tuple(sorted((a, b_prime, c_prime))) not in E: continue
                    if tuple(sorted((a_prime, b_prime, c_prime))) not in E: continue
                    return True
        return False

    def sanitize(E):
        """Remove triples that are part of a K(2,2,2)."""
        changed = True
        while changed:
            changed = False
            for T in list(E):
                if creates_conflict(E, T):
                    E.discard(T)
                    changed = True
                    break

    def greedy_fill(E):
        """Add any triple that doesn't create a conflict."""
        changed = True
        while changed:
            changed = False
            for T in all_triples:
                if T not in E:
                    if not creates_conflict(E, T):
                        E.add(T)
                        changed = True

    def local_search_swap(E, max_swaps=100):
        """Try to improve by swapping out a triple for a better one."""
        E_list = list(E)
        improved = True
        while improved and max_swaps > 0:
            improved = False
            for i in range(len(E_list)):
                if time.time() - start_time > timeout - 5:
                    return E
                T_out = E_list[i]
                # Try adding a random triple not in E
                T_in = random.choice(all_triples)
                if T_in in E:
                    continue
                # Check if removing T_out and adding T_in is valid
                E_new = set(E)
                E_new.discard(T_out)
                E_new.add(T_in)
                # Check validity: no K(2,2,2)
                valid = True
                for T in E_new:
                    if creates_conflict(E_new, T):
                        valid = False
                        break
                if valid:
                    E = E_new
                    E_list = list(E)
                    improved = True
                    max_swaps -= 1
                    break
        return E

    best_set = set()
    
    # Strategy: Multiple restarts with different initial densities
    for init_size in [300, 350, 400, 450, 500, 550, 600]:
        if time.time() - start_time > timeout:
            break
            
        for restart in range(15):
            if time.time() - start_time > timeout:
                break
                
            # Random initialization
            E = set(random.sample(all_triples, min(init_size, len(all_triples))))
            
            # Sanitize
            sanitize(E)
            
            # Fill
            greedy_fill(E)
            
            # Local search
            E = local_search_swap(E, max_swaps=50)
            
            # Record best
            if len(E) > len(best_set):
                best_set = set(E)
                
            if time.time() - start_time > timeout - 5:
                break

    # Final polish with more aggressive kicks
    if time.time() - start_time < timeout - 10:
        for _ in range(30):
            if time.time() - start_time > timeout - 5:
                break
            E = set(best_set)
            # Remove random triples
            if len(E) > 10:
                to_remove = random.sample(list(E), min(30, len(E)//4))
                for t in to_remove:
                    E.discard(t)
            
            sanitize(E)
            greedy_fill(E)
            E = local_search_swap(E, max_swaps=30)
            
            if len(E) > len(best_set):
                best_set = set(E)

    # Return as list of lists
    return [list(t) for t in best_set]
