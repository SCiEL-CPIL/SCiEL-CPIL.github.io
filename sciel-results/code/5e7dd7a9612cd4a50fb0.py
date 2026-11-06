"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
_N = 16
_M = 65
_DEADLINE = 148.0  # safety margin before 150s kill

def _make_bitsets():
    """Build K_{8,8} + edge (0,1) as list of Python int bitsets."""
    bits = [0] * _N
    for u in range(8):
        for v in range(8, 16):
            bits[u] |= (1 << v)
            bits[v] |= (1 << u)
    bits[0] |= (1 << 1)
    bits[1] |= (1 << 0)
    return bits

def _deg_from_bits(bits):
    return [bits[i].bit_count() for i in range(_N)]

def _score_bits(bits, deg):
    """Largest degree sum over all triangles using bitsets. Returns max_sum."""
    best = 0
    for u in range(_N):
        du = deg[u]
        for v in range(u + 1, _N):
            if bits[u] & (1 << v):
                dv = deg[v]
                # Common neighbors of u and v, only those > v to avoid duplicates
                common = bits[u] & bits[v] & ~((1 << (v + 1)) - 1)
                while common:
                    lsb = common & (-common)
                    w = lsb.bit_length() - 1
                    common ^= lsb
                    s = du + dv + deg[w]
                    if s > best:
                        best = s
    return best

def _edge_list_from_bits(bits):
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if bits[u] & (1 << v):
                edges.append([u, v])
    return edges

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Initial: K_{8,8} + (0,1)
    bits = _make_bitsets()
    deg = _deg_from_bits(bits)
    best_score = _score_bits(bits, deg)
    best_bits = bits[:]
    best_deg = deg[:]
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    def _swap_apply(bits, deg, u1, v1, u2, v2):
        """Remove edge (u1,v1), add edge (u2,v2)."""
        bits[u1] ^= (1 << v1)
        bits[v1] ^= (1 << u1)
        bits[u2] ^= (1 << v2)
        bits[v2] ^= (1 << u2)
        deg[u1] -= 1
        deg[v1] -= 1
        deg[u2] += 1
        deg[v2] += 1
    
    # Simulated annealing with bitset speed
    T = 8.0
    T_min = 0.05
    cooling = 0.9997
    
    stuck = 0
    best_stuck = 0
    steps = 0
    
    # Build edge/non-edge lists for random sampling
    def _get_edges(bits):
        el = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if bits[u] & (1 << v):
                    el.append((u, v))
        return el
    
    def _get_nonedges(bits):
        el = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if not (bits[u] & (1 << v)):
                    el.append((u, v))
        return el
    
    while time.time() - t0 < _DEADLINE - 20:
        # Sample random edge and non-edge
        edge_list = _get_edges(bits)
        nonedge_list = _get_nonedges(bits)
        
        if not edge_list or not nonedge_list:
            break
        
        e1 = random.choice(edge_list)
        e2 = random.choice(nonedge_list)
        
        u1, v1 = e1
        u2, v2 = e2
        
        _swap_apply(bits, deg, u1, v1, u2, v2)
        
        new_score = _score_bits(bits, deg)
        
        if new_score < best_score:
            best_score = new_score
            best_bits = bits[:]
            best_deg = deg[:]
            stuck = 0
        else:
            delta = new_score - best_score
            if delta > 0 and random.random() < math.exp(-delta / T):
                stuck += 1
            else:
                _swap_apply(bits, deg, u1, v1, u2, v2)
                stuck += 1
        
        T = max(T_min, T * cooling)
        steps += 1
        
        if stuck > 3000:
            bits = best_bits[:]
            deg = best_deg[:]
            T = 8.0
            stuck = 0
            best_stuck += 1
            if best_stuck > 30:
                break
    
    # Exhaustive local search polish from best state
    if time.time() - t0 < _DEADLINE - 5:
        bits = best_bits[:]
        deg = best_deg[:]
        improved = True
        passes = 0
        while improved and time.time() - t0 < _DEADLINE - 3 and passes < 50:
            improved = False
            passes += 1
            for e1 in all_pairs:
                if time.time() - t0 > _DEADLINE - 3:
                    break
                u1, v1 = e1
                if not (bits[u1] & (1 << v1)):
                    continue
                for e2 in all_pairs:
                    u2, v2 = e2
                    if bits[u2] & (1 << v2):
                        continue
                    _swap_apply(bits, deg, u1, v1, u2, v2)
                    s = _score_bits(bits, deg)
                    if s < best_score:
                        best_score = s
                        best_bits = bits[:]
                        best_deg = deg[:]
                        improved = True
                    else:
                        _swap_apply(bits, deg, u1, v1, u2, v2)
    
    edges = _edge_list_from_bits(best_bits)
    return {'n': _N, 'edges': edges}
# EVOLVE-BLOCK-END
