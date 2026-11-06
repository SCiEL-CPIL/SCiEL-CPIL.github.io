import numpy as np
import time
import itertools
from itertools import combinations
from math import comb

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def count_violations_fast(colors, M, nwords, quad_masks, quint_masks, quad_triples, quint_triples):
        """Count violations using bitset operations."""
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        for i in range(M):
            bit = np.uint64(1) << np.uint64(i & 63)
            w = i >> 6
            if colors[i] == 0:
                red_set[w] |= bit
            else:
                blue_set[w] |= bit
        
        r4 = np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1).sum()
        b5 = np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1).sum()
        return int(r4) + int(b5)
    
    def build_precomputed(n):
        """Precompute all structural data for n vertices."""
        M = comb(n, 3)
        triples = list(combinations(range(n), 3))
        tri_to_idx = {tri: i for i, tri in enumerate(triples)}
        
        quad_triples = []
        for quad in combinations(range(n), 4):
            idxs = [tri_to_idx[tuple(quad[j] for j in range(4) if j != i)] for i in range(4)]
            quad_triples.append(idxs)
        quint_triples = []
        for quint in combinations(range(n), 5):
            idxs = [tri_to_idx[t] for t in combinations(quint, 3)]
            quint_triples.append(idxs)
        
        quad_triples = np.array(quad_triples, dtype=np.int32)
        quint_triples = np.array(quint_triples, dtype=np.int32)
        
        nwords = (M + 63) // 64
        quad_masks = np.zeros((len(quad_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quad_triples):
            for t in tidxs:
                quad_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        quint_masks = np.zeros((len(quint_triples), nwords), dtype=np.uint64)
        for qidx, tidxs in enumerate(quint_triples):
            for t in tidxs:
                quint_masks[qidx, t >> 6] |= (np.uint64(1) << np.uint64(t & 63))
        
        return M, triples, quad_triples, quint_triples, nwords, quad_masks, quint_masks
    
    def simulated_annealing(n, time_limit, seed=0):
        """Simulated annealing with multi-flip moves on violating cliques."""
        M, triples, quad_triples, quint_triples, nwords, quad_masks, quint_masks = build_precomputed(n)
        
        rng = np.random.RandomState(seed)
        
        # Initialize with modular coloring
        colors = np.zeros(M, dtype=np.int8)
        for i, (a, b, c) in enumerate(triples):
            colors[i] = 0 if (a + b + c) % 3 == 0 else 1
        
        # Build initial bitsets
        red_set = np.zeros(nwords, dtype=np.uint64)
        blue_set = np.zeros(nwords, dtype=np.uint64)
        for i in range(M):
            bit = np.uint64(1) << np.uint64(i & 63)
            w = i >> 6
            if colors[i] == 0:
                red_set[w] |= bit
            else:
                blue_set[w] |= bit
        
        def get_violations():
            """Return lists of violating quad indices and quint indices."""
            viol_quads = np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]
            viol_quints = np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]
            return viol_quads, viol_quints
        
        def flip_triples(tidxs, new_color):
            """Flip a set of triples to new_color, update bitsets."""
            for t in tidxs:
                old = colors[t]
                if old == new_color:
                    continue
                bit = np.uint64(1) << np.uint64(t & 63)
                w = t >> 6
                if old == 0:
                    red_set[w] &= ~bit
                else:
                    blue_set[w] &= ~bit
                colors[t] = new_color
                if new_color == 0:
                    red_set[w] |= bit
                else:
                    blue_set[w] |= bit
        
        def count_viol():
            vq = np.where(np.all((quad_masks & red_set[None, :]) == quad_masks, axis=1))[0]
            vq5 = np.where(np.all((quint_masks & blue_set[None, :]) == quint_masks, axis=1))[0]
            return len(vq) + len(vq5)
        
        # Simulated annealing
        T0 = 10.0
        T_end = 0.01
        best_energy = count_viol()
        best_colors = colors.copy()
        
        step = 0
        total_steps = 200000
        
        for step in range(total_steps):
            if time.time() > start_time + time_limit:
                break
            
            T = T0 * (T_end / T0) ** (step / total_steps)
            
            viol_quads, viol_quints = get_violations()
            
            if len(viol_quads) + len(viol_quints) == 0:
                return colors.tolist()
            
            # Choose a move: either flip all red triples in a violating quad to blue,
            # or flip all blue triples in a violating quint to red, or single random flip
            move_type = rng.random()
            
            if move_type < 0.4 and len(viol_quads) > 0:
                # Flip red triples in a random violating quad to blue
                q = viol_quads[rng.randint(len(viol_quads))]
                tidxs = quad_triples[q]
                # Only flip those that are red
                to_flip = [t for t in tidxs if colors[t] == 0]
                if not to_flip:
                    to_flip = [int(t) for t in tidxs]
                    new_color = 1
                else:
                    new_color = 1
                old_energy = count_viol()
                flip_triples(to_flip, new_color)
                new_energy = count_viol()
                delta = new_energy - old_energy
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    pass  # accept
                else:
                    # revert
                    flip_triples(to_flip, 0)
                    # revert non-red ones back
                    for t in tidxs:
                        if t not in to_flip:
                            pass  # they weren't flipped
            elif move_type < 0.7 and len(viol_quints) > 0:
                # Flip blue triples in a random violating quint to red
                q = viol_quints[rng.randint(len(viol_quints))]
                tidxs = quint_triples[q]
                to_flip = [t for t in tidxs if colors[t] == 1]
                if not to_flip:
                    to_flip = [int(t) for t in tidxs]
                    new_color = 0
                else:
                    new_color = 0
                old_energy = count_viol()
                flip_triples(to_flip, new_color)
                new_energy = count_viol()
                delta = new_energy - old_energy
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    pass
                else:
                    flip_triples(to_flip, 1)
            else:
                # Single random flip
                t = int(rng.randint(0, M))
                new_color = 1 - colors[t]
                old_energy = count_viol()
                flip_triples([t], new_color)
                new_energy = count_viol()
                delta = new_energy - old_energy
                if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    pass
                else:
                    flip_triples([t], colors[t])  # revert
            
            if count_viol() < best_energy:
                best_energy = count_viol()
                best_colors = colors.copy()
        
        if best_energy == 0:
            return best_colors.tolist()
        return None
    
    # Fallback valid small coloring
    def fallback(n):
        triples = list(combinations(range(n), 3))
        return [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    best_n = 9
    best_colors = fallback(9)
    
    # Try increasing n with multiple restarts
    for n in range(10, 35):
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        per_n_budget = min(remaining - 3.0, 25.0)
        if per_n_budget <= 0:
            break
        n_restarts = 4
        per_restart = per_n_budget / n_restarts
        found = None
        for r in range(n_restarts):
            if time.time() > deadline - 3.0:
                break
            t_limit = min(per_restart, deadline - time.time() - 3.0)
            if t_limit <= 0.5:
                break
            res = simulated_annealing(n, t_limit, seed=n * 1000 + r * 17 + 3)
            if res is not None:
                found = res
                best_n = n
                best_colors = found
                break  # Found for this n, move to next n
        
    if best_n < 10:
        n = 9
        triples = list(combinations(range(n), 3))
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    
    return {'n': best_n, 'colors': best_colors}
