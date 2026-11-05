import time
import random

def construct():
    start = time.time()
    DEADLINE = 85
    N = 180
    
    # Precompute triples
    # (a, b, c) with a < b, a < c, b != c
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                if d == e:
                    continue
                b1 = a + d
                c1 = a + e
                if b1 <= N and c1 <= N and b1 != c1:
                    triples.append((a, b1, c1))
                b2 = a + e
                c2 = a + d
                if b2 <= N and c2 <= N and b2 != c2:
                    if (a, b2, c2) != (a, b1, c1):
                        triples.append((a, b2, c2))
    
    # Deduplicate
    seen = set()
    unique_triples = []
    for t in triples:
        # a is always smallest
        if t[0] < t[1] and t[0] < t[2]:
            fs = frozenset(t)
            if len(fs) == 3 and fs not in seen:
                seen.add(fs)
                unique_triples.append(tuple(sorted(fs)))

    # Identify elements that MUST be removed if we keep [91, 180]
    # An element a in [1, 90] is forced out if there is a triple (a, b, c) with b, c in [91, 180]
    forced_out = set()
    for a, b, c in unique_triples:
        if a <= 90 and b >= 91 and c >= 91:
            forced_out.add(a)
            
    # Elements in [1, 90] that are candidates
    candidates = [a for a in range(1, 91) if a not in forced_out]
    
    # Build constraints among candidates
    # Edge: (u, v) if there is a triple (u, v, w) or (w, u, v) etc where one is in [91, 180]
    # Actually, if (a, b, c) is a triple:
    # If b >= 91 and c in candidates: edge (a, c)
    # If c >= 91 and b in candidates: edge (a, b)
    # If b, c in candidates: hyperedge (a, b, c)
    
    edges = set()
    hyperedges = []
    
    for a, b, c in unique_triples:
        # a is smallest
        if a in forced_out:
            continue
        # b, c are larger
        b_in_cand = b in candidates
        c_in_cand = c in candidates
        b_large = b >= 91
        c_large = c >= 91
        
        if b_large and c_large:
            # Should have been in forced_out, but check just in case
            pass
        elif b_large and c_in_cand:
            edges.add((a, c))
        elif c_large and b_in_cand:
            edges.add((a, b))
        elif b_in_cand and c_in_cand:
            hyperedges.append((a, b, c))
        # If one is not in candidates and not large? 
        # candidates covers [1, 90] \ forced_out.
        # If b < 91 and b not in candidates, it's in forced_out.
        # So if b is in forced_out, the triple is already broken.
        
    # Now we want to select a subset of 'candidates' such that:
    # 1. No edge (u, v) has both u, v selected.
    # 2. No hyperedge (u, v, w) has all three selected.
    
    # This is Max Independent Set with some hyperedges.
    # We can use local search.
    
    # Initial solution: all candidates
    S_small = set(candidates)
    
    # Fix violations
    def fix(S):
        changed = True
        while changed:
            changed = False
            # Check edges
            for u, v in edges:
                if u in S and v in S:
                    # Remove one with higher degree? Or just random?
                    # Let's remove the one with more constraints
                    # Count violations involving u and v
                    # Simple: remove v
                    S.discard(v)
                    changed = True
            # Check hyperedges
            for u, v, w in hyperedges:
                if u in S and v in S and w in S:
                    S.discard(w)
                    changed = True
        return S
    
    S_small = fix(S_small)
    
    # Hill Climbing
    rng = random.Random(42)
    best_S_small = set(S_small)
    best_size = len(best_S_small)
    
    for _ in range(500):
        if time.time() > DEADLINE:
            break
        # Try adding a candidate
        excluded = [c for c in candidates if c not in best_S_small]
        if not excluded:
            break
        improved = False
        for x in excluded:
            trial = set(best_S_small)
            trial.add(x)
            fixed = fix(trial)
            if len(fixed) > best_size:
                best_S_small = fixed
                best_size = len(best_S_small)
                improved = True
        if not improved:
            # Try swaps
            for _ in range(10):
                if time.time() > DEADLINE:
                    break
                if not best_S_small:
                    break
                x_out = rng.choice(list(best_S_small))
                x_in = rng.choice(excluded)
                trial = set(best_S_small)
                trial.discard(x_out)
                trial.add(x_in)
                fixed = fix(trial)
                if len(fixed) > best_size:
                    best_S_small = fixed
                    best_size = len(best_S_small)
                    improved = True
        if not improved:
            break
            
    # Final Set
    S_final = set(range(91, 181)) | best_S_small
    
    # Final verification and cleanup
    in_final = set(S_final)
    for t in unique_triples:
        if all(x in in_final for x in t):
            # Remove one element
            # Prefer removing the smallest (a)
            S_final.discard(t[0])
            in_final.discard(t[0])
            
    if not S_final:
        S_final = {1}
        
    return list(S_final)
