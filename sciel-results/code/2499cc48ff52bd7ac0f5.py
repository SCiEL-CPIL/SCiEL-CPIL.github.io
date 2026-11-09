def _cayley_zq(q, girth_target, k):
    """Try Cayley graphs on Z_q with k generators."""
    if k % 2 == 1:
        # For odd k: S = {±a, ±b, c} where c != 0, c != ±a, c != ±b
        for a in range(1, min(q, 30)):
            for b in range(1, min(q, 30)):
                if b == a or b == q - a:
                    continue
                for c in range(1, min(q, 30)):
                    if c in (a, b, q-a, q-b, 0):
                        continue
                    S = {a, b, c, (q-a) % q, (q-b) % q}
                    if len(S) != k:
                        continue
                    
                    n = q
                    adj = [[] for _ in range(n)]
                    edges = []
                    for x in range(n):
                        for s in S:
                            y = (x + s) % n
                            adj[x].append(y)
                            if x < y:
                                edges.append((x, y))
                    
                    if _girth_ok(adj, n, girth_target):
                        return n, edges
    return None
