<<<<<<< SEARCH
                    elif mode == 'big_then_split':
                        # Try big first; if conflict, split into two halves
                        cc_in_neigh = set()
                        for i in range(d):
                            for j in range(i + 1, d):
                                a, b = neigh[i], neigh[j]
                                e = (a, b) if a < b else (b, a)
                                cc_in_neigh.add(e)
                        if not (cc_in_neigh & used_cc):
                            cliques.append([v] + neigh)
                            used_cc |= cc_in_neigh
                        else:
                            # Try splitting into two groups
                            mid = d // 2
                            g1 = neigh[:mid]
                            g2 = neigh[mid:]
                            # Check g1
                            cc1 = set()
                            for i in range(len(g1)):
                                for j in range(i + 1, len(g1)):
                                    a, b = g1[i], g1[j]
                                    e = (a, b) if a < b else (b, a)
                                    cc1.add(e)
                            if not (cc1 & used_cc) and len(g1) >= 2:
                                cliques.append([v] + g1)
                                used_cc |= cc1
                            elif len(g1) >= 2:
                                for u in g1:
                                    cliques.append([v, u])
                            elif len(g1) == 1:
                                cliques.append([v, g1[0]])
                            # Check g2
                            cc2 = set()
                            for i in range(len(g2)):
                                for j in range(i + 1, len(g2)):
                                    a, b = g2[i], g2[j]
                                    e = (a, b) if a < b else (b, a)
                                    cc2.add(e)
                            if not (cc2 & used_cc) and len(g2) >= 2:
                                cliques.append([v] + g2)
                                used_cc |= cc2
                            elif len(g2) >= 2:
                                for u in g2:
                                    cliques.append([v, u])
                            elif len(g2) == 1:
                                cliques.append([v, g2[0]])
=======
                    elif mode == 'big_then_split':
                        # Try big first; if conflict, use greedy_pack for this vertex
                        cc_in_neigh = set()
                        for i in range(d):
                            for j in range(i + 1, d):
                                a, b = neigh[i], neigh[j]
                                e = (a, b) if a < b else (b, a)
                                cc_in_neigh.add(e)
                        if not (cc_in_neigh & used_cc):
                            cliques.append([v] + neigh)
                            used_cc |= cc_in_neigh
                        else:
                            # Greedy pack for this vertex
                            remaining_neigh = list(neigh)
                            groups = []
                            while remaining_neigh:
                                group = [remaining_neigh.pop(0)]
                                changed = True
                                while changed:
                                    changed = False
                                    for idx in range(len(remaining_neigh)):
                                        cand = remaining_neigh[idx]
                                        ok = all(
                                            (cand, g) if cand < g else (g, cand) not in used_cc
                                            for g in group
                                        )
                                        if ok:
                                            group.append(cand)
                                            remaining_neigh.pop(idx)
                                            changed = True
                                            break
                                groups.append(group)
                            for group in groups:
                                if len(group) == 1:
                                    cliques.append([v, group[0]])
                                else:
                                    cliques.append([v] + group)
                                    for i2 in range(len(group)):
                                        for j2 in range(i2 + 1, len(group)):
                                            a, b = group[i2], group[j2]
                                            e = (a, b) if a < b else (b, a)
                                            used_cc.add(e)
>>>>>>> REPLACE

