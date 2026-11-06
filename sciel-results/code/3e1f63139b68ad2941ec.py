"""
Maximize triples on 16 points avoiding K(2,2,2) transversal.
Uses pattern-based counting for fast validity checks during local search.
"""
import time
import random
import itertools
from collections import defaultdict

def _precompute_patterns():
    """
    Precompute all K(2,2,2) patterns. Each pattern is a frozenset of 8 triples.
    Also build a map: triple -> list of pattern indices containing it.
    """
    patterns = []
    triple_to_patterns = defaultdict(list)
    
    for six in itertools.combinations(range(16), 6):
        a, b, c, d, e, f = six
        pairings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in pairings:
            pattern_triples = set()
            for x in (0, 1):
                for y in (0, 1):
                    for z in (0, 1):
                        t = tuple(sorted((p1[x], p2[y], p3[z])))
                        pattern_triples.add(t)
            pattern_frozen = frozenset(pattern_triples)
            patterns.append(pattern_frozen)
            for t in pattern_triples:
                triple_to_patterns[t].append(len(patterns) - 1)
    
    return patterns, dict(triple_to_patterns)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute patterns
    patterns, triple_to_patterns = _precompute_patterns()
    n_patterns = len(patterns)
    
    # Pattern counts: how many of the 8 triples in each pattern are selected
    pattern_counts = [0] * n_patterns
    
    def add_triple(t):
        """Add triple t, update pattern counts. Returns True if still valid."""
        for pi in triple_to_patterns[t]:
            pattern_counts[pi] += 1
            if pattern_counts[pi] == 8:
                # Invalid! Rollback
                for pj in triple_to_patterns[t]:
                    pattern_counts[pj] -= 1
                return False
        return True
    
    def remove_triple(t):
        """Remove triple t, update pattern counts."""
        for pi in triple_to_patterns[t]:
            pattern_counts[pi] -= 1
    
    def is_valid():
        """Check if no pattern has count 8."""
        return all(c < 8 for c in pattern_counts)
    
    all_triples = list(itertools.combinations(range(16), 3))
    best_triples = []
    best_len = 0
    
    # Multiple randomized greedy constructions
    for seed in range(100):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        selected = set()
        pattern_counts_local = [0] * n_patterns
        
        for t in shuffled:
            # Check if adding t keeps all patterns < 8
            ok = True
            for pi in triple_to_patterns[t]:
                if pattern_counts_local[pi] + 1 >= 8:
                    ok = False
                    break
            if ok:
                for pi in triple_to_patterns[t]:
                    pattern_counts_local[pi] += 1
                selected.add(t)
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_triples = list(selected)
    
    # Local search: try to add triples not currently selected
    if time.time() < deadline - 5 and best_triples:
        selected = set(best_triples)
        pattern_counts_local = [0] * n_patterns
        for t in selected:
            for pi in triple_to_patterns[t]:
                pattern_counts_local[pi] += 1
        
        not_selected = [t for t in all_triples if t not in selected]
        rng = random.Random(999)
        
        for _ in range(200000):
            if time.time() > deadline:
                break
            if not not_selected:
                break
            t = rng.choice(not_selected)
            # Check if we can add t
            ok = True
            for pi in triple_to_patterns[t]:
                if pattern_counts_local[pi] + 1 >= 8:
                    ok = False
                    break
            if ok:
                for pi in triple_to_patterns[t]:
                    pattern_counts_local[pi] += 1
                selected.add(t)
                not_selected.remove(t)
                if len(selected) > best_len:
                    best_len = len(selected)
                    best_triples = list(selected)
    
    return best_triples
