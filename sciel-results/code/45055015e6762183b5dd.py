import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Precompute all 4-subsets of [1,16] as bitmasks
    all_subsets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_subsets.append(mask)
    
    n = len(all_subsets)  # 1820
    
    # Precompute pairwise intersections
    # INTER[i][j] = all_subsets[i] & all_subsets[j]
    INTER = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = all_subsets[i]
        row = INTER[i]
        for j in range(i + 1, n):
            v = mi & all_subsets[j]
            row[j] = v
            INTER[j][i] = v
    
    def greedy_conflict_weighted(order, deadline):
        """
        Greedy that at each step picks the candidate with minimum conflict count.
        Conflict count for candidate c = number of pairs (a,b) in family such that
        INTER[a][b] == INTER[a][c] == INTER[b][c]
        
        We maintain for each candidate c a running conflict count.
        When we add a new set s, we update conflict counts for all remaining candidates.
        """
        chosen = []
        chosen_set = set()
        remaining = list(order)
        
        # conflict_count[c] = number of existing pairs that would form sunflower with c
        conflict_count = {c: 0 for c in remaining}
        
        t0 = time.time()
        
        while remaining:
            if time.time() - t0 > deadline - 3.0:
                break
            
            # Find candidate with minimum conflict count
            min_conf = float('inf')
            best_c = None
            for c in remaining:
                cc = conflict_count[c]
                if cc < min_conf:
                    min_conf = cc
                    best_c = c
                    if cc == 0:
                        break  # Can't do better than 0
            
            if best_c is None:
                break
            
            # Add best_c to family
            chosen.append(best_c)
            chosen_set.add(best_c)
            remaining.remove(best_c)
            del conflict_count[best_c]
            
            # Update conflict counts for all remaining candidates
            # For each remaining candidate r, check if adding best_c creates new sunflower pairs
            # A sunflower is formed when there exist a, b in chosen (including best_c) such that
            # INTER[a][b] == INTER[a][r] == INTER[b][r]
            # Since best_c is new, new sunflowers involving best_c:
            # For each a in chosen (excluding best_c):
            #   Let v = INTER[a][best_c]
            #   For each remaining r:
            #     If INTER[a][r] == v and INTER[best_c][r] == v, then (a, best_c, r) is a sunflower
            #     But wait - that's not quite right. The sunflower is (a, b, c) where all three pairwise
            #     intersections are equal. So for new sunflowers involving best_c, we need pairs (a, best_c)
            #     where a is in chosen, and check if INTER[a][best_c] == INTER[a][r] == INTER[best_c][r]
            
            # Actually, the conflict_count[r] should count pairs (a,b) in chosen such that
            # INTER[a][b] == INTER[a][r] == INTER[b][r]
            # When we add best_c, new pairs involving best_c are (best_c, a) for a in chosen[:-1]
            # For each such pair (best_c, a), and for each remaining r:
            #   If INTER[best_c][a] == INTER[best_c][r] == INTER[a][r], then this pair contributes to conflict_count[r]
            
            for a in chosen[:-1]:  # All previously chosen sets
                v_ab = INTER[best_c][a]
                for r in remaining:
                    if INTER[best_c][r] == v_ab and INTER[a][r] == v_ab:
                        conflict_count[r] += 1
        
        return chosen
    
    def local_search(family, deadline):
        """
        Local search: try to add sets that were rejected, or do swaps.
        Simple approach: repeatedly try to add any set not in family.
        If stuck, try removing one set and adding two.
        """
        family_set = set(family)
        t0 = time.time()
        
        improved = True
        while improved and time.time() - t0 < deadline - 3.0:
            improved = False
            
            # Try to add any set not in family
            for s in range(n):
                if s in family_set:
                    continue
                if time.time() - t0 > deadline - 3.0:
                    break
                
                # Check if s can be added
                ok = True
                for i in range(len(family)):
                    a = family[i]
                    inter_as = INTER[a][s]
                    for j in range(i + 1, len(family)):
                        b = family[j]
                        if INTER[a][b] == inter_as and INTER[b][s] == inter_as:
                            ok = False
                            break
                    if not ok:
                        break
                
                if ok:
                    family.append(s)
                    family_set.add(s)
                    improved = True
                    break  # Restart the scan after adding
        
        return family
    
    best_family = []
    
    # Phase 1: Multiple greedy restarts with different random orders
    rng = random.Random(12345)
    n_restarts = 0
    while time.time() - start_time < deadline - 10.0:
        order = list(range(n))
        rng.shuffle(order)
        
        fam = greedy_conflict_weighted(order, deadline)
        
        if len(fam) > len(best_family):
            best_family = fam[:]
        
        n_restarts += 1
        
        # If we're running long, reduce restarts
        if n_restarts > 50 and time.time() - start_time > deadline - 20.0:
            break
    
    # Phase 2: Local search on best family
    if time.time() - start_time < deadline - 5.0:
        best_family = local_search(best_family, deadline)
    
    # Phase 3: More local search with different starting points
    if time.time() - start_time < deadline - 5.0:
        for _ in range(3):
            if time.time() - start_time > deadline - 5.0:
                break
            order = list(range(n))
            rng.shuffle(order)
            fam = greedy_conflict_weighted(order, deadline)
            fam = local_search(fam, deadline)
            if len(fam) > len(best_family):
                best_family = fam[:]
    
    # Convert masks back to sorted lists of elements
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found family of size {len(res)}")
