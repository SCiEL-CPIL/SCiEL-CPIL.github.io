                    critical_set = set()
                    for s in subs1:
                        if coverage_count[s] == 1:
                            critical_set.add(int(s))
                    for s in subs2:
                        if coverage_count[s] == 1:
                            critical_set.add(int(s))
                    set1 = block_triple_sets[b1]
                    set2 = block_triple_sets[b2]
                    for s in set1 & set2:
                        if coverage_count[s] == 2:
                            critical_set.add(int(s))
