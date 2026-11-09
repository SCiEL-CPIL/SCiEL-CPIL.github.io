            if random.random() < 0.02:
                # Try to add a few random non-edges
                for _ in range(10):
                    u2 = random.randint(0, n_vertices - 1)
                    v2 = random.randint(0, n_vertices - 1)
                    if u2 == v2 or (u2, v2) in edges:
                        continue
                    if not creates_c6(u2, v2, adj):
                        edges.add((u2, v2))
                        edges.add((v2, u2))
                        adj[u2].add(v2)
                        adj[v2].add(u2)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = set(edges)
                        break
