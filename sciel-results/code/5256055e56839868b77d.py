"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    # Precompute pair membership for linearity check
    # For each pair of points, which triples contain that pair
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(t[i], t[j]), max(t[i], t[j]))
                pair_to_triples[pair].append(t)
    
    def is_linear(triple_set, new_triple):
        """Check if adding new_triple maintains linearity."""
        for i in range(3):
            for j in range(i+1, 3):
                pair = (min(new_triple[i], new_triple[j]), max(new_triple[i], new_triple[j]))
                for existing in pair_to_triples[pair]:
                    if existing in triple_set:
                        return False
        return True
    
    def compute_independence_number(triples):
        """Compute the independence number of the hypergraph."""
        if not triples:
            return n
        # A set S is independent iff no edge is fully contained in S
        # Equivalent: V\S is a vertex cover
        # Find minimum vertex cover, then alpha = n - tau
        # Check all subsets by size, from 0 upward, find smallest vertex cover
        triples_set = set(triples)
        # For efficiency, precompute for each point which triples it's in
        point_to_triples = defaultdict(set)
        for t in triples_set:
            for p in t:
                point_to_triples[p].add(t)
        
        # Find minimum vertex cover
        best_tau = n
        # Try all subsets
        for r in range(0, n+1):
            if r >= best_tau:
                break
            found = False
            for subset in itertools.combinations(range(n), r):
                subset_set = set(subset)
                # Check if subset is a vertex cover
                is_cover = True
                for t in triples_set:
                    if not (t[0] in subset_set or t[1] in subset_set or t[2] in subset_set):
                        is_cover = False
                        break
                if is_cover:
                    best_tau = r
                    found = True
                    break
            if found:
                break
        return n - best_tau
    
    def greedy_build(rng):
        """Greedy randomized build of a linear hypergraph."""
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        triples = []
        triple_set = set()
        for t in shuffled:
            if is_linear(triple_set, t):
                triples.append(t)
                triple_set.add(t)
        return triples
    
    best_alpha = n
    best_triples = None
    
    # Phase 1: Multiple greedy restarts
    rng = random.Random(42)
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        triples = greedy_build(rng)
        alpha = compute_independence_number(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            # If we get very low, we can stop early
            if best_alpha <= 4:
                break
    
    # Phase 2: Local search - try to improve by swapping triples
    if best_triples is not None and time.time() < deadline:
        current_triples = best_triples[:]
        current_alpha = best_alpha
        triple_set = set(current_triples)
        
        # Try removing one triple and adding a different one
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Pick a random existing triple to remove
            idx = rng.randrange(len(current_triples))
            removed = current_triples[idx]
            current_triples.pop(idx)
            triple_set.remove(removed)
            
            # Try to add a random triple
            candidates = []
            for _ in range(50):
                t = rng.choice(all_triples)
                if t not in triple_set and is_linear(triple_set, t):
                    candidates.append(t)
            
            if candidates:
                new_triple = rng.choice(candidates)
                current_triples.append(new_triple)
                triple_set.add(new_triple)
                alpha = compute_independence_number(current_triples)
                if alpha < current_alpha:
                    current_alpha = alpha
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = current_triples[:]
                        if best_alpha <= 4:
                            break
                else:
                    # Revert
                    current_triples.remove(new_triple)
                    triple_set.remove(new_triple)
                    current_triples.insert(idx, removed)
                    triple_set.add(removed)
    
    # Return the best result
    if best_triples is not None:
        return [[t[0], t[1], t[2]] for t in best_triples]
    else:
        # Fallback: return the original simple construction
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
# EVOLVE-BLOCK-END
