"""Improved sum-free subset finder using spectral bisection on Schur graph."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                if S[i] + S[j] in S_set:
                    return False
        return True

    def spectral_bisection(values):
        """Use spectral bisection on the Schur graph to find a large independent set."""
        n = len(values)
        if n == 0:
            return set()
        if n == 1:
            return set(values)

        val_to_idx = {v: i for i, v in enumerate(values)}
        vset = set(values)

        # Build adjacency matrix for Schur graph:
        # Edge between i and j if:
        #   1) values[i] + values[j] is in vset (and the sum's index != i, j)
        #   2) 2*values[i] == values[j] (i.e., i + i = j)
        #   3) 2*values[j] == values[i]
        # Actually, for sum-free, we need: no a, b, c in S with a + b = c.
        # The conflict graph: vertex v conflicts with w if v + w = z for some z in values,
        # OR 2v = w for some w in values, OR 2w = v.
        # But actually the constraint is on the SET: we cannot have a, b, c all in S.
        # So we can model this as: we want a subset S such that for all a, b in S, a+b not in S.
        # This is equivalent to: S is an independent set in a hypergraph, but for the graph version:
        # Create a graph where edge (i,j) exists if there exists k such that values[i]+values[j]=values[k]
        # (including k=i or k=j, i.e., 2*values[i]=values[j]).
        # Then S is sum-free iff S is an independent set in this graph.
        # Wait, that's not quite right. If a+b=c, then all three are in S. The edge (a,b) 
        # "forbids" having both a and b if c is also present. But we're choosing the subset.
        # 
        # Actually, a cleaner formulation: define graph G where vertices are the elements,
        # and we put an edge between x and y (x != y) if x + y is in the original set.
        # Also, we put a self-loop on x if 2x is in the original set (meaning we can't pick x).
        # Wait, no. If we pick x and 2x is also picked, that violates sum-free (x + x = 2x).
        # 
        # Let me think again. S is sum-free if for all a, b in S (a=b allowed), a+b not in S.
        # So: if a in S, then 2a not in S.
        # And: if a, b in S (a != b), then a+b not in S.
        #
        # Graph model: vertex for each element. Edge (i, j) for i != j if values[i] + values[j] is in vset.
        # Self-loop on i if 2*values[i] is in vset.
        # Then S is sum-free iff: no vertex with self-loop is in S, and no edge has both endpoints in S.
        # 
        # Hmm, but that's not quite a standard graph independent set because of self-loops.
        # Vertices with self-loops simply cannot be in S.
        # For the remaining vertices, S must be an independent set in the graph.
        
        # First, remove vertices with self-loops (2*values[i] in vset)
        has_self_loop = np.zeros(n, dtype=bool)
        for i in range(n):
            if 2 * values[i] in vset:
                has_self_loop[i] = True

        # Build adjacency among remaining vertices
        # Edge (i, j), i < j, i,j not self-loop, if values[i] + values[j] in vset
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            if has_self_loop[i]:
                continue
            for j in range(i + 1, n):
                if has_self_loop[j]:
                    continue
                if values[i] + values[j] in vset:
                    adj[i, j] = 1
                    adj[j, i] = 1

        # Fiedler vector (second eigenvector) for bisection
        # Use numpy's eigvalsh for symmetric matrices
        try:
            eigvals, eigvecs = np.linalg.eigh(adj.astype(float))
            fiedler = eigvecs[:, 1]  # second smallest eigenvalue's eigenvector
        except:
            # Fallback: greedy
            fiedler = np.random.randn(n)

        # Partition: positive side vs negative side
        pos_mask = fiedler > 0
        neg_mask = ~pos_mask

        # Try both partitions and also refine
        def refine(S_indices):
            """Greedy refinement: try adding back removed elements and swapping."""
            S_set = set(values[i] for i in S_indices)
            changed = True
            # Try adding elements not in S
            for i in range(n):
                if i in S_set_idx or has_self_loop[i]:
                    continue
                # Check if adding values[i] violates sum-free
                ok = True
                for v in S_set:
                    if values[i] + v in S_set or v + values[i] in S_set:
                        if values[i] + v in S_set:
                            ok = False
                            break
                    if 2 * values[i] in S_set:
                        ok = False
                        break
                # Also need: values[i] + values[i] = 2*values[i] not in S_set
                if 2 * values[i] in S_set:
                    ok = False
                if ok:
                    S_set.add(values[i])
            return S_set

        # Get candidate sets from both sides
        pos_set = set(values[i] for i in range(n) if pos_mask[i] and not has_self_loop[i])
        neg_set = set(values[i] for i in range(n) if neg_mask[i] and not has_self_loop[i])

        # Also try the other orientation
        # Fiedler vector sign is arbitrary, so both are valid candidates

        best = max(pos_set, neg_set, key=len)

        # Greedy improvement: try to add elements from the other side
        other_set = neg_set if best is pos_set else pos_set
        improved = set(best)
        for v in sorted(other_set):
            # Try adding v
            ok = True
            for u in list(improved):
                if v + u in improved:
                    ok = False
                    break
            if 2 * v in improved:
                ok = False
            if ok:
                improved.add(v)

        best = max(best, improved, key=len)

        return best

    def greedy_sum_free(values):
        """Simple greedy: sort by absolute value, add if possible."""
        S = set()
        for v in sorted(values, key=abs):
            # Check if adding v violates sum-free
            ok = True
            for u in list(S):
                if v + u in S:
                    ok = False
                    break
            if 2 * v in S:
                ok = False
            if ok:
                S.add(v)
        return S

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odd numbers if out of time
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue

        vals = instance(size, seed)
        
        # Try multiple approaches and keep the best
        candidates = []
        
        # 1. Odd numbers baseline
        odd_set = set(x for x in vals if x % 2)
        candidates.append(odd_set)
        
        # 2. Spectral bisection
        try:
            spec_set = spectral_bisection(vals)
            candidates.append(spec_set)
        except Exception:
            pass
        
        # 3. Greedy by various orderings
        for order in ['abs_asc', 'abs_desc', 'asc', 'desc']:
            if order == 'abs_asc':
                ordered = sorted(vals, key=abs)
            elif order == 'abs_desc':
                ordered = sorted(vals, key=abs, reverse=True)
            elif order == 'asc':
                ordered = sorted(vals)
            else:
                ordered = sorted(vals, reverse=True)
            
            S = set()
            for v in ordered:
                ok = True
                for u in list(S):
                    if v + u in S:
                        ok = False
                        break
                if 2 * v in S:
                    ok = False
                if ok:
                    S.add(v)
            candidates.append(S)
        
        # 4. Randomized greedy with multiple restarts
        import random
        for restart in range(10):
            if time.time() > deadline:
                break
            rng = random.Random(seed + restart)
            shuffled = list(vals)
            rng.shuffle(shuffled)
            S = set()
            for v in shuffled:
                ok = True
                for u in list(S):
                    if v + u in S:
                        ok = False
                        break
                if 2 * v in S:
                    ok = False
                if ok:
                    S.add(v)
            candidates.append(S)
        
        # Keep the largest valid candidate
        best = max(candidates, key=lambda S: (len(S), is_sum_free(S)))
        
        # Verify it's sum-free; if not, fall back
        if not is_sum_free(best):
            best = odd_set  # odd numbers are always sum-free
        
        results.append(sorted(best))
    
    return results
# EVOLVE-BLOCK-END
