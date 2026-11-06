import random
import itertools

def construct():
    n = 16
    points = list(range(n))
    full_mask = (1 << n) - 1
    
    def fast_calc_alpha(edges):
        if not edges:
            return n
        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
        is_indep = [True] * (1 << n)
        for em in edge_masks:
            complement = (~em) & full_mask
            sub = complement
            while True:
                s = em | sub
                is_indep[s] = False
                if sub == 0:
                    break
                sub = (sub - 1) & complement
        max_alpha = 0
        for i in range(1 << n):
            if is_indep[i]:
                pc = bin(i).count('1')
                if pc > max_alpha:
                    max_alpha = pc
                    if max_alpha == n:
                        break
        return max_alpha

    def is_linear(edge, existing_edges):
        edge_set = set(edge)
        for e in existing_edges:
            if len(edge_set & set(e)) > 1:
                return False
        return True

    all_triples = list(itertools.combinations(points, 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflicts
    conflicts = [set() for _ in range(len(all_triples))]
    for i in range(len(all_triples)):
        ti = all_triples[i]
        si = set(ti)
        for j in range(i + 1, len(all_triples)):
            tj = all_triples[j]
            sj = set(tj)
            if len(si & sj) > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Precompute 6-sets and their triples
    all_6sets = list(itertools.combinations(points, 6))
    # s6_to_triples: list of triple indices
    s6_to_triples = []
    for s6 in all_6sets:
        s6_set = set(s6)
        indices = []
        for t in all_triples:
            if set(t).issubset(s6_set):
                indices.append(triple_index[t])
        s6_to_triples.append(indices)

    def try_cover_6sets():
        E = []
        available = set(range(len(all_triples)))
        uncovered = set(range(len(all_6sets))) # indices of 6-sets
        
        while uncovered:
            # Pick a random uncovered 6-set
            s6_idx = random.choice(list(uncovered))
            opts = [t for t in s6_to_triples[s6_idx] if t in available]
            
            if not opts:
                return None
            
            # Pick a triple from opts
            # Prefer one that covers more uncovered 6-sets
            # Score: number of uncovered 6-sets containing t
            # We can approximate by just picking random from opts to be fast
            # Or pick the one with highest degree in the remaining graph?
            # Let's just pick random for speed, restart if fail.
            t_idx = random.choice(opts)
            
            E.append(all_triples[t_idx])
            
            # Update available
            available -= conflicts[t_idx]
            available.remove(t_idx)
            
            # Update uncovered
            # Remove all 6-sets that contain t_idx
            # We can iterate all uncovered and check if t_idx is in s6_to_triples[s6]
            # This is O(|uncovered| * 20).
            new_uncovered = set()
            for u_idx in uncovered:
                if t_idx not in s6_to_triples[u_idx]:
                    new_uncovered.add(u_idx)
            uncovered = new_uncovered
            
        return E

    best_E = None
    best_alpha = 16
    
    # Try CSP heuristic
    for _ in range(100):
        res = try_cover_6sets()
        if res:
            alpha = fast_calc_alpha(res)
            if alpha <= 5:
                best_E = res
                break
            elif alpha < best_alpha:
                best_alpha = alpha
                best_E = res

    if best_E is None:
        # Fallback greedy
        for restart in range(100):
            E = []
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for edge in shuffled:
                if is_linear(edge, E):
                    E.append(edge)
            alpha = fast_calc_alpha(E)
            if alpha < best_alpha:
                best_alpha = alpha
                best_E = E[:]

    return [list(e) for e in best_E]
