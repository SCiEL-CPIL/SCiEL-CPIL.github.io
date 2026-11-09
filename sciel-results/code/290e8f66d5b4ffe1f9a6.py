import random
import time
import math

_START = time.time()
_LIMIT = 110.0  # Safety margin

def _compute_alpha_fast(edges):
    """
    Compute independence number of hypergraph with 16 vertices.
    edges: list of ints (bitmasks)
    """
    n = 16
    all_vertices_mask = (1 << n) - 1
    
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in range(n):
            if e & (1 << v):
                vertex_edges[v].append(i)
                
    best = [0]
    
    # Precompute popcount for masks? No, use bit_count.
    
    def is_valid_add(v, S):
        for e_idx in vertex_edges[v]:
            e = edges[e_idx]
            others = e ^ (1 << v)
            if (others & S) == others:
                return False
        return True

    def get_new_avail(S, v_bit, old_avail):
        new_S = S | v_bit
        new_avail = old_avail ^ v_bit
        # Filter new_avail
        final_avail = 0
        # Iterate bits in new_avail
        temp = new_avail
        while temp:
            u_bit = temp & -temp
            u = u_bit.bit_length() - 1
            temp ^= u_bit
            
            blocked = False
            for e_idx in vertex_edges[u]:
                e = edges[e_idx]
                # Check if e \ {u} is subset of new_S
                # e ^ u_bit is the rest of the edge
                rest = e ^ u_bit
                if (rest & new_S) == rest:
                    blocked = True
                    break
            
            if not blocked:
                final_avail |= u_bit
                
        return final_avail

    def dfs(S, avail):
        # Prune
        if S.bit_count() + avail.bit_count() <= best[0]:
            return
        
        if avail == 0:
            if S.bit_count() > best[0]:
                best[0] = S.bit_count()
            return
            
        # Pick a vertex
        # Heuristic: Pick vertex with max degree in current subgraph?
        # For speed, just pick lowest bit.
        v_bit = avail & -avail
        v = v_bit.bit_length() - 1
        
        # Option 1: Exclude v
        dfs(S, avail ^ v_bit)
        
        # Option 2: Include v
        if is_valid_add(v, S):
            new_avail = get_new_avail(S, v_bit, avail)
            dfs(S | v_bit, new_avail)
            
    dfs(0, all_vertices_mask)
    return best[0]

def _is_valid_triple_set(triples):
    """Check if the set of triples is linear (any two share at most 1 point)."""
    n = len(triples)
    for i in range(n):
        a, b, c = triples[i]
        for j in range(i + 1, n):
            x, y, z = triples[j]
            inter = 0
            if a == x or a == y or a == z: inter += 1
            if b == x or b == y or b == z: inter += 1
            if c == x or c == y or c == z: inter += 1
            if inter > 1:
                return False
    return True

def construct():
    rng = random.Random(42)
    
    # Precompute all possible triples as bitmasks
    all_triples_masks = []
    all_triples_tuples = []
    for i in range(16):
        for j in range(i + 1, 16):
            for k in range(j + 1, 16):
                mask = (1 << i) | (1 << j) | (1 << k)
                all_triples_masks.append(mask)
                all_triples_tuples.append((i, j, k))
                
    def get_used_pairs(triples):
        pairs = set()
        for a, b, c in triples:
            pairs.add((a, b))
            pairs.add((a, c))
            pairs.add((b, c))
        return pairs

    def is_conflict(triple, used_pairs):
        a, b, c = triple
        if (a, b) in used_pairs or (a, c) in used_pairs or (b, c) in used_pairs:
            return True
        return False

    def greedy_init():
        triples = []
        used_pairs = set()
        remaining = all_triples_tuples[:]
        rng.shuffle(remaining)
        
        for t in remaining:
            if not is_conflict(t, used_pairs):
                triples.append(t)
                a, b, c = t
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
                
        return triples

    def simulate_annealing(init_triples, time_budget):
        current_triples = list(init_triples)
        current_masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples ]
        current_alpha = _compute_alpha_fast(current_masks)
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        T = 1.0
        cooling = 0.999
        start_time = time.time()
        
        while (time.time() - start_time) < time_budget:
            if time.time() - _START > _LIMIT:
                break
                
            move_type = rng.random()
            new_triples = None
            new_alpha = None
            delta = 0
            
            if move_type < 0.3:
                # Add a triple
                t = rng.choice(all_triples_tuples)
                if t in current_triples:
                    continue
                used_pairs = get_used_pairs(current_triples)
                if not is_conflict(t, used_pairs):
                    new_triples = current_triples + [t]
                    new_masks = current_masks + [ (1<<t[0])|(1<<t[1])|(1<<t[2]) ]
                    new_alpha = _compute_alpha_fast(new_masks)
                    delta = new_alpha - current_alpha
            elif move_type < 0.6:
                # Remove a triple
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                new_triples = current_triples[:idx] + current_triples[idx+1:]
                new_masks = current_masks[:idx] + current_masks[idx+1:]
                new_alpha = _compute_alpha_fast(new_masks)
                delta = new_alpha - current_alpha
            else:
                # Swap
                if not current_triples:
                    continue
                idx = rng.randrange(len(current_triples))
                t_remove = current_triples[idx]
                t_add = rng.choice(all_triples_tuples)
                if t_add == t_remove:
                    continue
                if t_add in current_triples:
                    continue
                
                temp_triples = current_triples[:idx] + current_triples[idx+1:]
                used_pairs = get_used_pairs(temp_triples)
                
                if not is_conflict(t_add, used_pairs):
                    new_triples = temp_triples + [t_add]
                    new_masks = current_masks[:idx] + current_masks[idx+1:] + [ (1<<t_add[0])|(1<<t_add[1])|(1<<t_add[2]) ]
                    new_alpha = _compute_alpha_fast(new_masks)
                    delta = new_alpha - current_alpha

            if new_triples is None:
                continue

            if delta < 0:
                current_triples = new_triples
                current_masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples ]
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
            else:
                if T > 0:
                    if delta == 0:
                        if rng.random() < 0.5:
                            current_triples = new_triples
                            current_masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples ]
                            current_alpha = new_alpha
                    else:
                        prob = math.exp(-delta / T)
                        if rng.random() < prob:
                            current_triples = new_triples
                            current_masks = [ (1<<a)|(1<<b)|(1<<c) for a,b,c in current_triples ]
                            current_alpha = new_alpha
            
            T *= cooling
            if T < 0.01:
                T = 0.01

        return best_triples, best_alpha

    best_global_alpha = 16
    best_global_triples = None
    
    start_time = time.time()
    restarts = 0
    while (time.time() - start_time) < (_LIMIT - 10):
        init_triples = greedy_init()
        if not init_triples:
            continue
        
        budget = min(5.0, max(1.0, _LIMIT - 15 - (time.time() - start_time)))
        
        opt_triples, alpha = simulate_annealing(init_triples, budget)
        
        if alpha < best_global_alpha:
            best_global_alpha = alpha
            best_global_triples = opt_triples
            
        restarts += 1
        if best_global_alpha <= 5:
            break

    if best_global_triples is None:
        best_global_triples = greedy_init()
        if not _is_valid_triple_set(best_global_triples):
            best_global_triples = []
            
    return best_global_triples

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
    print(f"Alpha: {_compute_alpha_fast([(1<<a)|(1<<b)|(1<<c) for a,b,c in res])}")
