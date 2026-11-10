"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def edge_idx(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def build_c4_list():
    c4s = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    c4s.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(a, d)))
                    c4s.append((edge_idx(a, b), edge_idx(b, d), edge_idx(c, d), edge_idx(a, c)))
                    c4s.append((edge_idx(a, c), edge_idx(b, c), edge_idx(b, d), edge_idx(a, d)))
    return c4s

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 70
    
    c4s = build_c4_list()
    num_c4s = len(c4s)
    
    # Build edge -> list of c4 indices
    edge_to_c4s = [[] for _ in range(NUM_EDGES)]
    for ci, c4 in enumerate(c4s):
        for ei in c4:
            edge_to_c4s[ei].append(ci)
    
    # Initial valid coloring: bipartite 2-coloring
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = edge_idx(i, j)
            if (i < 8 and j < 8) or (i >= 8 and j >= 8):
                colors[idx] = 0
            else:
                colors[idx] = 1
    
    best_colors = colors[:]
    best_count = 2
    
    rng = random.Random(12345)
    
    # Precompute for fast c4 validity check
    def c4_valid(c4):
        v0 = colors[c4[0]]
        v1 = colors[c4[1]]
        v2 = colors[c4[2]]
        v3 = colors[c4[3]]
        # Not all four distinct
        return not (v0 != v1 and v0 != v2 and v0 != v3 and v1 != v2 and v1 != v3 and v2 != v3)
    
    # Local search
    for it in range(200000):
        if time.time() > deadline:
            break
        
        ei = rng.randint(0, NUM_EDGES - 1)
        old = colors[ei]
        
        # Choose new color: bias toward unused
        used_set = set(colors)
        if len(used_set) < MAX_COLOR and rng.random() < 0.6:
            # Pick an unused color
            unused = [c for c in range(MAX_COLOR) if c not in used_set]
            new_c = rng.choice(unused)
        else:
            new_c = rng.randint(0, MAX_COLOR - 1)
        
        if new_c == old:
            continue
        
        # Check validity of all c4s containing edge ei
        valid = True
        for ci in edge_to_c4s[ei]:
            c4 = c4s[ci]
            colors[ei] = new_c
            if not c4_valid(c4):
                valid = False
                break
        if not valid:
            colors[ei] = old
            continue
        
        colors[ei] = new_c
        cnt = len(set(colors))
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
        
        # Try a swap move
        if rng.random() < 0.4:
            ei2 = rng.randint(0, NUM_EDGES - 1)
            if ei2 == ei:
                continue
            c1, c2 = colors[ei], colors[ei2]
            if c1 == c2:
                continue
            
            # Collect affected c4s
            aff = set(edge_to_c4s[ei])
            aff.update(edge_to_c4s[ei2])
            
            ok = True
            saved = {}
            for ci in aff:
                c4 = c4s[ci]
                for e in c4:
                    if e not in saved:
                        saved[e] = colors[e]
                colors[ei] = c2
                colors[ei2] = c1
                if not c4_valid(c4):
                    ok = False
                    break
            # Restore
            for e, v in saved.items():
                colors[e] = v
            
            if ok:
                colors[ei] = c2
                colors[ei2] = c1
                cnt = len(set(colors))
                if cnt > best_count:
                    best_count = cnt
                    best_colors = colors[:]
    
    return best_colors
# EVOLVE-BLOCK-END
