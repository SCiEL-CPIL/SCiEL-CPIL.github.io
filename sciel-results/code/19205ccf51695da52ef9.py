            # Pruning: Check for isolated unvisited vertices
            for w in range(n):
                if not visited[w]:
                    # Count neighbors in unvisited set
                    cnt = 0
                    for nb in adj[w]:
                        if not visited[nb]:
                            cnt += 1
                    if cnt == 0:
                        # Must be connected to current u
                        if u not in adj[w]:
                            return None
