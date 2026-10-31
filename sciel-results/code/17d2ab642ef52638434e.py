                        for rest in itertools.combinations(block, t - 1):
                            tsub = tuple(sorted(rest + (x,)))
                            if tsub in mapping:
                                i = mapping[tsub]
                                if i in newly_uncovered and coverage_count[i] == 0:
                                    gain += 1
