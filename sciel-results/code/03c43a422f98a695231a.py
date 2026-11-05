def has_short_path(u, v, max_len, adj):
    if u == v:
        return True
    dist_u = {u: 0}
    dist_v = {v: 0}
    queue_u = deque([u])
    queue_v = deque([v])
    while queue_u or queue_v:
        if len(queue_u) <= len(queue_v):
            x = queue_u.popleft()
            dx = dist_u[x]
            if dx >= max_len:
                continue
            for y in adj[x]:
                if y not in dist_u:
                    dist_u[y] = dx + 1
                    if y in dist_v:
                        if dx + 1 + dist_v[y] <= max_len:
                            return True
                    if dx + 1 < max_len:
                        queue_u.append(y)
        else:
            x = queue_v.popleft()
            dx = dist_v[x]
            if dx >= max_len:
                continue
            for y in adj[x]:
                if y not in dist_v:
                    dist_v[y] = dx + 1
                    if y in dist_u:
                        if dx + 1 + dist_u[y] <= max_len:
                            return True
                    if dx + 1 < max_len:
                        queue_v.append(y)
    return False
