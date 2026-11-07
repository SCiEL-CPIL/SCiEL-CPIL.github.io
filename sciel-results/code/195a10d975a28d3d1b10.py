                for u in adj[v]:
                    if u in unassigned_set:
                        if not remove_color(u, color):
                            # Color wasn't in domain, so no change, no conflict from this specific removal
                            pass 
                        else:
                            # Color was removed
                            if len(domains[u]) == 0:
                                conflict = True
                                break
