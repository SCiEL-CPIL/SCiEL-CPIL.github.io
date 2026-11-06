"""Search for linear triple systems on 16 points with small independence number."""
import time
import random
import sys
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _check_linearity(triples):
    """Check that any two triples share at most one point."""
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            common = set(triples[i]) & set(triples[j])
            if len(common) > 1:
                return False
    return True


def _independence_number(triples):
    """Find the largest subset of points containing no complete triple."""
    points = set()
    for t in triples:
        points.update(t)
    n = len(points)
    if n == 0:
        return 0
    triple_set = set()
    for t in triples:
        triple_set.add(frozenset(t))
    
    # Check all subsets from largest to smallest
    # For n=16, check subsets of size 15, 14, ...
    all_points = sorted(points)
    for size in range(n, 0, -1):
        for subset in combinations(all_points, size):
            subset_set = set(subset)
            has_triple = False
            for t in triples:
                if set(t) <= subset_set:
                    has_triple = True
                    break
            if not has_triple:
                return size
    return 0


def _independence_number_fast(triples):
    """Faster independence number using bitmasks for n=16."""
    points = set()
    for t in triples:
        points.update(t)
    n = len(points)
    if n == 0:
        return 0
    
    # Map points to 0..n-1
    point_list = sorted(points)
    point_to_idx = {p: i for i, p in enumerate(point_list)}
    
    # Represent each triple as a bitmask
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << point_to_idx[p])
        triple_masks.append(mask)
    
    full_mask = (1 << n) - 1
    
    # Check subsets from largest to smallest
    for size in range(n, 0, -1):
        # Generate all subsets of given size
        for subset in combinations(range(n), size):
            subset_mask = 0
            for p in subset:
                subset_mask |= (1 << p)
            # Check if any triple is fully contained
            contained = False
            for tm in triple_masks:
                if (tm & subset_mask) == tm:
                    contained = True
                    break
            if not contained:
                return size
    return 0


def _try_complete(triples, all_triples, time_limit):
    """Try to extend partial triples to a full system."""
    start = time.time()
    triples = [list(t) for t in triples]
    
    def backtrack(idx, used_pairs):
        if time.time() > time_limit:
            return None
        if idx == len(all_triples):
            return triples[:]
        
        tri = all_triples[idx]
        # Check if this triple conflicts
        tri_set = set(tri)
        for existing in triples:
            common = tri_set & set(existing)
            if len(common) > 1:
                return None
        
        triples.append(tri)
        result = backtrack(idx + 1, used_pairs)
        if result is not None:
            return result
        triples.pop()
        return None
    
    return backtrack(0, set())


def _greedy_construct(random_state):
    """Construct a linear triple system using greedy + local search."""
    random.seed(random_state)
    points = list(range(16))
    
    # Generate all possible triples
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    
    triples = []
    for tri in all_triples:
        valid = True
        for existing in triples:
            if len(set(tri) & set(existing)) > 1:
                valid = False
                break
        if valid:
            triples.append(list(tri))
        if len(triples) >= 56:  # Steiner triple system has 56 triples for v=16? No, (16*15)/6 = 40
            break
    
    return triples


def _local_search(triples, random_state, time_limit):
    """Improve the system by trying to reduce independence number."""
    random.seed(random_state)
    best_triples = [list(t) for t in triples]
    best_score = _independence_number_fast(best_triples)
    
    while time.time() < time_limit:
        # Try random modifications
        if random.random() < 0.3 and len(best_triples) > 0:
            # Remove a random triple
            idx = random.randint(0, len(best_triples) - 1)
            removed = best_triples.pop(idx)
            # Try to add a new triple
            candidates = []
            for a in range(16):
                for b in range(a + 1, 16):
                    for c in range(b + 1, 16):
                        new_tri = [a, b, c]
                        valid = True
                        for existing in best_triples:
                            if len(set(new_tri) & set(existing)) > 1:
                                valid = False
                                break
                        if valid:
                            candidates.append(new_tri)
            if candidates:
                new_tri = random.choice(candidates)
                best_triples.append(new_tri)
                score = _independence_number_fast(best_triples)
                if score < best_score:
                    best_score = score
                else:
                    # Revert
                    best_triples.remove(new_tri)
                    best_triples.append(removed)
        else:
            # Try swapping a triple
            if len(best_triples) > 0:
                idx = random.randint(0, len(best_triples) - 1)
                old_tri = best_triples[idx]
                candidates = []
                for a in range(16):
                    for b in range(a + 1, 16):
                        for c in range(b + 1, 16):
                            new_tri = [a, b, c]
                            valid = True
                            for j, existing in enumerate(best_triples):
                                if j == idx:
                                    continue
                                if len(set(new_tri) & set(existing)) > 1:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(new_tri)
                if candidates:
                    new_tri = random.choice(candidates)
                    best_triples[idx] = new_tri
                    score = _independence_number_fast(best_triples)
                    if score < best_score:
                        best_score = score
                    else:
                        best_triples[idx] = old_tri
    
    return best_triples, best_score


def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    time_limit = start_time + 135.0  # 15s safety margin
    
    best_triples = None
    best_score = 16  # worst case
    
    # Phase 1: Multiple greedy constructions with different random seeds
    for seed in range(50):
        if time.time() > time_limit:
            break
        triples = _greedy_construct(seed)
        if not triples:
            continue
        score = _independence_number_fast(triples)
        if score < best_score:
            best_score = score
            best_triples = [list(t) for t in triples]
    
    # Phase 2: Local search to improve
    if best_triples is not None:
        remaining_time = time_limit - time.time()
        if remaining_time > 5:
            # Run local search
            triples, score = _local_search(best_triples, 42, time_limit)
            if score < best_score:
                best_score = score
                best_triples = triples
    
    # Final deadline check
    if time.time() > time_limit - 2:
        # Return what we have
        if best_triples is not None:
            return best_triples
        # Fallback: seed program
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    if best_triples is not None:
        return best_triples
    
    # Fallback: seed program
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
