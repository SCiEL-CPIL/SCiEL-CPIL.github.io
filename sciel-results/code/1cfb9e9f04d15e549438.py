import time
import itertools
import random

N = 16
ALL_TRIPLES = list(itertools.combinations(range(N), 3))
ALL_QUADS = list(itertools.combinations(range(N), 4))

# Precompute pairs for each triple
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    pairs = set()
    for i in range(3):
        for j in range(i+1, 3):
            pairs.add((t[i], t[j]))
    TRIPLE_PAIRS.append(pairs)

def _can_add(triples, new_triple, used_pairs_set):
    for p in _pairs_of(new_triple):
        if p in used_pairs_set:
            return False
    return True

def _pairs_of(triple):
    return set(itertools.combinations(triple, 2))

def _used_pairs(triples):
    s = set()
    for t in triples:
        s |= _pairs_of(t)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_triples = None
    best_alpha = N + 1
    
    # Precompute quad coverage for K=4 specifically to speed up the most likely case
    # But let's keep it general inside the solver.
    
    def solve_for_alpha(K, time_limit):
        """
        Backtracking search for a linear hypergraph on N vertices 
        such that alpha <= K.
        """
        # Precompute K-subsets and coverage
        K_subsets = list(itertools.combinations(range(N), K))
        num_ksubsets = len(K_subsets)
        
        # Map: triple_idx -> set of K_subset_indices it covers
        TRIPLE_COVERS_KSUBSETS = [set() for _ in range(len(ALL_TRIPLES))]
        for i, ks in enumerate(K_subsets):
            ks_set = set(ks)
            for t_idx in range(len(ALL_TRIPLES)):
                if all(x in ks_set for x in ALL_TRIPLES[t_idx]):
                    TRIPLE_COVERS_KSUBSETS[t_idx].add(i)
        
        # Initial state
        # Symmetry breaking: Fix first triple to (0, 1, 2)
        idx_012 = ALL_TRIPLES.index((0, 1, 2))
        init_triples_idx = [idx_012]
        init_used_triple_set = {idx_012}
        init_used_pairs = set(TRIPLE_PAIRS[idx_012])
        init_uncovered = set(range(num_ksubsets)) - TRIPLE_COVERS_KSUBSETS[idx_012]
        
        def bt(triples_idx, used_triple_set, used_pairs, uncovered_set):
            nonlocal best_triples, best_alpha
            if time.time() > time_limit:
                return False
            
            if not uncovered_set:
                best_triples = [ALL_TRIPLES[i] for i in triples_idx]
                best_alpha = K
                return True
            
            # Generate candidates
            candidates = []
            for t_idx in range(len(ALL_TRIPLES)):
                if t_idx in used_triple_set:
                    continue
                if not TRIPLE_PAIRS[t_idx].isdisjoint(used_pairs):
                    continue
                
                # Count coverage
                cov = len(TRIPLE_COVERS_KSUBSETS[t_idx] & uncovered_set)
                if cov > 0:
                    candidates.append((cov, t_idx))
            
            if not candidates:
                return False
            
            # Sort by coverage descending
            candidates.sort(reverse=True)
            
            for _, t_idx in candidates:
                new_triples_idx = triples_idx + [t_idx]
                new_used_triple_set = used_triple_set | {t_idx}
                new_used_pairs = used_pairs | TRIPLE_PAIRS[t_idx]
                new_uncovered_set = uncovered_set - TRIPLE_COVERS_KSUBSETS[t_idx]
                
                if bt(new_triples_idx, new_used_triple_set, new_used_pairs, new_uncovered_set):
                    return True
            
            return False
        
        if bt(init_triples_idx, init_used_triple_set, init_used_pairs, init_uncovered):
            return True
        return False

    # Try to find a solution with alpha <= 4
    if solve_for_alpha(4, deadline):
        return best_triples
    
    # Try alpha <= 5
    if solve_for_alpha(5, deadline):
        return best_triples
        
    # Fallback to random/greedy if backtracking fails or times out
    random.seed(42)
    triples = []
    used = set()
    indices = list(range(len(ALL_TRIPLES)))
    random.shuffle(indices)
    for idx in indices:
        t = ALL_TRIPLES[idx]
        if _can_add(triples, t, used):
            triples.append(t)
            used |= _pairs_of(t)
    return triples

