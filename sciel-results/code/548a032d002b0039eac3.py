<<<<<<< SEARCH
        else:
            # For larger trees, use a more sophisticated approach
            # Try multiple random permutations and heuristics
            deg = [0] * k
            for a, b in edges:
                deg[a] += 1
                deg[b] += 1
            vorder = list(range(k))
            vorder.sort(key=lambda v: -deg[v])
            
            mapping = {}
            
            def dfs3(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    for a, b in edges:
                        e = frozenset((mapping[a], mapping[b]))
                        if e in used:
                            return False
                    return True
                v = vorder[pos]
                # Try vertices in a heuristic order
                candidates = list(range(n))
                if w in mapping.values():
                    continue
                for w in candidates:
                    if w in mapping.values():
                        continue
                    ok = True
                    for a, b in edges:
                        if a == v and b in mapping:
                            e = frozenset((w, mapping[b]))
                            if e in used:
                                ok = False
                                break
                        elif b == v and a in mapping:
                            e = frozenset((w, mapping[a]))
                            if e in used:
                                ok = False
                                break
                    if not ok:
                        continue
                    mapping[v] = w
                    if dfs3(pos + 1):
                        return True
                    del mapping[v]
                return False
            
            if dfs3(0):
                res = list(mapping[i] for i in range(k))
                result[tree_idx] = res
                for a, b in edges:
                    used_edges.add(frozenset((res[a], res[b])))
                backtrack(idx + 1, count + len(edges))
                for a, b in edges:
                    used_edges.discard(frozenset((res[a], res[b])))
                result[tree_idx] = None
                return
            else:
                result[tree_idx] = None
                backtrack(idx + 1, count)
                return
=======
        else:
            # For larger trees, use a more sophisticated approach
            # Try multiple random permutations and heuristics
            deg = [0] * k
            for a, b in edges:
                deg[a] += 1
                deg[b] += 1
            vorder = list(range(k))
            vorder.sort(key=lambda v: -deg[v])
            
            mapping = {}
            
            def dfs3(pos):
                if time.time() > deadline:
                    return False
                if pos == k:
                    for a, b in edges:
                        e = frozenset((mapping[a], mapping[b]))
                        if e in used:
                            return False
                    return True
                v = vorder[pos]
                # Try vertices in a heuristic order
                for w in range(n):
                    if w in mapping.values():
                        continue
                    ok = True
                    for a, b in edges:
                        if a == v and b in mapping:
                            e = frozenset((w, mapping[b]))
                            if e in used:
                                ok = False
                                break
                        elif b == v and a in mapping:
                            e = frozenset((w, mapping[a]))
                            if e in used:
                                ok = False
                                break
                    if not ok:
                        continue
                    mapping[v] = w
                    if dfs3(pos + 1):
                        return True
                    del mapping[v]
                return False
            
            if dfs3(0):
                res = list(mapping[i] for i in range(k))
                result[tree_idx] = res
                for a, b in edges:
                    used_edges.add(frozenset((res[a], res[b])))
                backtrack(idx + 1, count + len(edges))
                for a, b in edges:
                    used_edges.discard(frozenset((res[a], res[b])))
                result[tree_idx] = None
                return
            else:
                result[tree_idx] = None
                backtrack(idx + 1, count)
                return
>>>>>>> REPLACE
