"""Rewritten program using AC-3 and Backtracking with Domain Copying."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: 120s limit, leave 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use bitmasks for domains for faster operations
        # Colors are 0..4, so we use bits 0..4
        doms = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            doms[i] = mask
        
        if any(d == 0 for d in doms):
            return None

        # AC-3
        q = [(u, v) for u in range(n) for v in adj[u]]
        def revise(i, j):
            # If doms[j] has exactly one color, remove it from doms[i] if present
            dj = doms[j]
            if dj and (dj & (dj - 1)) == 0: # dj is power of 2 (single color)
                if doms[i] & dj:
                    doms[i] &= ~dj
                    return True
            return False

        head = 0
        while head < len(q):
            i, j = q[head]
            head += 1
            if revise(i, j):
                if doms[i] == 0:
                    return None
                for k in adj[i]:
                    if k != j:
                        q.append((k, i))
        
        if any(d == 0 for d in doms):
            return None

        def bt(doms, assign):
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            # MRV: Find unassigned vertex with smallest domain size
            min_sz = float('inf')
            min_v = -1
            for i in range(n):
                if i not in assign:
                    sz = doms[i].bit_count()
                    if sz < min_sz:
                        min_sz = sz
                        min_v = i
                        if sz == 0:
                            return None
                        if sz == 1:
                            break
            if min_v == -1:
                return assign # Solution found

            # LVC: Sort values by least constraining
            # Get list of colors in doms[min_v]
            mask = doms[min_v]
            vals = []
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                vals.append(c)
                mask ^= lsb
            
            # Heuristic: count conflicts
            def count_conflicts(c):
                bit = 1 << c
                cnt = 0
                for nb in adj[min_v]:
                    if nb not in assign:
                        if doms[nb] & bit:
                            cnt += 1
                return cnt
            
            vals.sort(key=count_conflicts)
            
            for c in vals:
                bit = 1 << c
                
                # Check validity against assigned neighbors
                ok = True
                for nb in adj[min_v]:
                    if nb in assign:
                        if assign[nb] == c:
                            ok = False
                            break
                if not ok:
                    continue
                
                # Propagate
                new_doms = doms[:]
                new_doms[min_v] = bit
                for nb in adj[min_v]:
                    if nb not in assign:
                        if new_doms[nb] & bit:
                            new_doms[nb] &= ~bit
                            if new_doms[nb] == 0:
                                ok = False
                                break
                if not ok:
                    continue
                
                new_assign = dict(assign)
                new_assign[min_v] = c
                
                res = bt(new_doms, new_assign)
                if res is not None:
                    return res
            return None

        try:
            sol_dict = bt(doms, {})
            if sol_dict:
                return [int(sol_dict[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Generate n to keep structure
            # We need n. We can estimate or just generate.
            # Generating is fast.
            n, _, _ = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            out.append([0] * n)
        else:
            out.append([int(c) for c in sol])
            
    return out
# EVOLVE-BLOCK-END
