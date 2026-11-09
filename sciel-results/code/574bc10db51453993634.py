    def try_merge(cliques, free_edges):
            changed = True
            while changed:
                if time.time() > deadline:
                    break
                changed = False
                for i in range(len(cliques)):
                    ci = cliques[i]
                    for j in range(i + 1, len(cliques)):
                        cj = cliques[j]
                        
                        # Check if union is a clique and edges are free
                        is_valid = True
                        # Check cross edges
                        # Optimization: if they share a vertex, only check edges between the non-shared parts
                        shared = set(ci) & set(cj)
                        if len(shared) > 1:
                            # Can't merge if they share > 1 vertex?
                            # If they share > 1 vertex, they are not a partition?
                            # The input `cliques` is a partition, so they are disjoint.
                            # So shared is empty.
                            pass
                        
                        # Since they are disjoint, check all pairs u in ci, v in cj
                        # But we can optimize:
                        # If both are C-only, check all cross edges in free_edges.
                        # If one has I, the other must be C-only or share the I vertex (impossible if disjoint).
                        # Wait, if they are disjoint, they cannot share an I vertex.
                        # So if one has an I vertex, the other must be C-only?
                        # No, they can both have I vertices, but different ones.
                        # If $v_1 \in ci \cap I$ and $v_2 \in cj \cap I$, then $v_1, v_2$ are not adjacent.
                        # So union is not a clique.
                        # So valid merges are:
                        # 1. Both C-only.
                        # 2. One C-only, one I-clique (with I vertex v).
                        #    In this case, the I vertex v must be adjacent to all C in the C-only clique.
                        #    And the C-C edges between the C-parts must be free.
                        
                        # Let's just check all pairs. M is small.
                        for u in ci:
                            for v in cj:
                                if u == v: continue
                                # Check adjacency in original graph
                                # We don't have original_adj here.
                                # But we know the structure.
                                # If u, v in C: adjacent.
                                # If u in C, v in I: adjacent if u in N[v].
                                # If u in I, v in C: adjacent if v in N[u].
                                # If u in I, v in I: NOT adjacent.
                                
                                if u < k and v < k:
                                    # C-C edge
                                    e = (min(u,v), max(u,v))
                                    if e not in free_edges:
                                        is_valid = False
                                        break
                                elif u < k and v >= k:
                                    # C-I edge
                                    # Must exist in original graph
                                    # And it is covered by the clique.
                                    # Is it "free"?
                                    # C-I edges are NEVER in free_edges (free_edges is C-C residual).
                                    # C-I edges are covered by I-cliques.
                                    # If we merge a C-only clique with an I-clique,
                                    # the C-I edges between them are covered by the new clique.
                                    # Were they covered before?
                                    # The C-only clique did not cover C-I edges.
                                    # The I-clique covered C-I edges for its C-part.
                                    # The cross C-I edges are NEW to the cover?
                                    # No, the I-clique $K_2$ covers edges $(u, v)$ for $u \in S_2$.
                                    # The C-only clique $K_1$ covers edges $(a, b)$ for $a,b \in K_1$.
                                    # The cross edges are $(u, v)$ with $u \in K_1, v \in I$ (the vertex in $K_2$).
                                    # These edges are in the original graph.
                                    # Are they covered by any other clique?
                                    # They are C-I edges.
                                    # C-I edges are covered by I-cliques.
                                    # Specifically, edge $(u, v)$ is covered by the clique containing $v$ and $u$.
                                    # In $K_2$, $v$ is present, but $u$ is in $K_1$ (disjoint).
                                    # So $(u, v)$ is NOT covered by $K_2$.
                                    # Is it covered by any other clique?
                                    # It must be, because every edge is covered.
                                    # If it is covered by another clique, we cannot cover it again.
                                    # So we can ONLY merge if the cross C-I edges are NOT covered by other cliques.
                                    # But they MUST be covered by some clique.
                                    # So we can NEVER merge a C-only clique with an I-clique?
                                    # Unless the C-only clique was formed from residual edges, and the C-I edges were... wait.
                                    # C-I edges are never residual. They are all covered by I-cliques.
                                    # So any C-I edge is already covered.
                                    # If we merge, we cover it again.
                                    # So NO merging between C-only and I-cliques is allowed.
                                    
                                    # What about two I-cliques?
                                    # $K_1 = \{v_1\} \cup S_1$, $K_2 = \{v_2\} \cup S_2$.
                                    # Cross edges:
                                    # $v_1, v_2$ (I-I): Not adjacent. Cannot merge.
                                    # So NO merging between I-cliques is allowed.
                                    
                                    # Conclusion: We can ONLY merge C-only cliques.
                                    # And for C-only cliques $K_1, K_2$, we can merge them iff all cross C-C edges are in `free_edges`.
                                    
                                    # This simplifies things greatly!
                                    pass
                                else:
                                    # I-I or other
                                    is_valid = False
                                    break
                            if not is_valid:
                                break
                        
                        if is_valid:
                            # Merge
                            merged = list(set(ci) | set(cj))
                            cliques = [c for idx, c in enumerate(cliques) if idx != i and idx != j]
                            cliques.append(merged)
                            changed = True
                            break
                    if changed:
                        break
            return cliques
