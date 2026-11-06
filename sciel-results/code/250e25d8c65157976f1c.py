import numpy as np
import time
import random
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(POINTS)
ALL_POINTS_MASK = (1 << N_POINTS) - 1

# Precompute squared distances
coords = np.array(POINTS, dtype=np.int32)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
D = np.sum(diff ** 2, axis=2)

# Precompute IsoMask[a, b]
IsoMask = [[0 for _ in range(N_POINTS)] for _ in range(N_POINTS)]

for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        d_ab = D[a, b]
        mask = 0
        for c in range(N_POINTS):
            if c == a or c == b:
                continue
            d_ac = D[a, c]
            d_bc = D[b, c]
            if d_ac == d_ab or d_bc == d_ab or d_ac == d_bc:
                mask |= (1 << c)
        IsoMask[a, b] = mask
        IsoMask[b, a] = mask

def construct():
    start_time = time.time()
    deadline = 300.0
    
    best_set = None
    best_size = N_POINTS + 1
    
    rng = random.Random(42)
    
    def greedy_build():
        S = []
        S_mask = 0
        Forbidden = 0
        
        # Random start
        start_idx = rng.randint(0, N_POINTS - 1)
        S.append(start_idx)
        S_mask |= (1 << start_idx)
        
        while True:
            if (Forbidden | S_mask) == ALL_POINTS_MASK:
                break
            
            candidates_mask = ALL_POINTS_MASK & ~S_mask & ~Forbidden
            
            if candidates_mask == 0:
                break
            
            best_p = -1
            best_score = -1
            best_new_forbidden = 0
            
            c = candidates_mask
            while c:
                lsb = c & -c
                p = lsb.bit_length() - 1
                c ^= lsb
                
                new_forbidden = Forbidden
                for a in S:
                    new_forbidden |= IsoMask[a, p]
                
                score = (new_forbidden & ~Forbidden).bit_count()
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_new_forbidden = new_forbidden
                
            if best_p == -1:
                break
            
            S.append(best_p)
            S_mask |= (1 << best_p)
            Forbidden = best_new_forbidden
            
        return S

    while time.time() - start_time < deadline:
        S = greedy_build()
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = [POINTS[i] for i in S]
            
            if best_size <= 5:
                break

    if best_set is None:
        best_set = [POINTS[0]]
        
    return best_set
