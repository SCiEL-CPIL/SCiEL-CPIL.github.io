"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _random_walk(values, edges, steps=20000, seed=42):
    n = len(values)
    m = len(edges)
    best_vals = list(values)
    best_edges = [list(e) for e in edges]
    best_score = _score(best_vals, best_edges)
    rng = random.Random(seed)
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    for _ in range(steps):
        move_type = rng.random()
        if move_type < 0.3:
            # Perturb values
            idx = rng.randrange(n)
            old_val = best_vals[idx]
            # Try a few new values
            new_val = old_val + rng.randint(-50, 50)
            if 1 <= new_val <= 10000 and new_val != old_val:
                # Check distinctness
                if new_val not in best_vals:
                    best_vals[idx] = new_val
                    s = _score(best_vals, best_edges)
                    if s > best_score:
                        best_vals[idx] = old_val
                    else:
                        if s < best_score:
                            best_score = s
        else:
            # Perturb edges: swap an edge
            if m < 1:
                continue
            # Pick an existing edge to remove and a new edge to add
            existing_edge = best_edges[rng.randrange(m)]
            # Pick a random non-edge
            all_pairs = list(itertools.combinations(range(n), 2))
            new_edge = all_pairs[rng.randrange(len(all_pairs))]
            if new_edge == tuple(existing_edge) or (tuple(sorted(new_edge)) == tuple(sorted(existing_edge))):
                continue
            # Remove existing, add new
            best_edges.remove(existing_edge)
            best_edges.append(list(new_edge))
            s = _score(best_vals, best_edges)
            if s > best_score:
                best_edges.remove(list(new_edge))
                best_edges.append(existing_edge)
            else:
                if s < best_score:
                    best_score = s
    return best_vals, best_edges, best_score

def construct():
    t0 = time.time()
    deadline = 75.0  # safety margin
    
    # Start with a simple AP and random edges
    values = list(range(1, 15))
    all_pairs = list(itertools.combinations(range(14), 2))
    rng = random.Random(123)
    edges = rng.sample(all_pairs, 35)
    edges = [list(e) for e in edges]
    
    best_vals, best_edges, best_score = values, edges, _score(values, edges)
    
    # Do a few random restarts with different value sets and edge sets
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate a new value set: random distinct positive integers
        v = set()
        while len(v) < 14:
            v.add(rng.randint(1, 10000))
        v_list = list(v)
        rng.shuffle(v_list)
        
        # Generate random edges
        e_set = set(rng.sample(all_pairs, 35))
        e_list = [list(x) for x in e_set]
        
        cur_score = _score(v_list, e_list)
        if cur_score < best_score:
            best_score = cur_score
            best_vals = v_list
            best_edges = e_list
        
        # Local search
        if time.time() > deadline:
            break
        vals, edges2, sc2 = _random_walk(v_list, e_list, steps=2000, seed=restart)
        if sc2 < best_score:
            best_score = sc2
            best_vals = vals
            best_edges = edges2
    
    # Final: ensure types are correct
    result = {'values': list(best_vals), 'edges': [list(e) for e in best_edges]}
    return result
# EVOLVE-BLOCK-END
