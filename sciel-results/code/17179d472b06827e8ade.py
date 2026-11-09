import random
from itertools import combinations

def construct():
    # Generate all 4-subsets of {1..16} as bitmasks
    # 1..16 maps to bits 0..15
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    best_family = []
    best_size = 0
    
    # Number of restarts
    # Estimate time per restart:
    # ~1820 candidates * ~200 chosen * (dict ops)
    # Let's run 200 restarts to be safe within 60s.
    for _ in range(200):
        random.shuffle(all_sets)
        chosen = []
        for S in all_sets:
            # Check compatibility
            groups = {}
            compatible = True
            for A in chosen:
                K = A & S
                if K in groups:
                    groups[K].append(A)
                else:
                    groups[K] = [A]
            
            # Check conflicts within groups
            for K, members in groups.items():
                if len(members) > 1:
                    for i in range(len(members)):
                        Ai = members[i]
                        for j in range(i + 1, len(members)):
                            Bj = members[j]
                            if (Ai & Bj) == K:
                                compatible = False
                                break
                        if not compatible:
                            break
                if not compatible:
                    break
            
            if compatible:
                chosen.append(S)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen
            
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
