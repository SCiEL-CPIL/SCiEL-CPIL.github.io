                if best_gain > 0 and best_r is not None:
                    S.discard(best_r)
                    # Re-verify additions against the live set S
                    # We sort by some heuristic? No, just try them.
                    # To be safe, we check each v against the current S.
                    # If a v fails, we skip it (we might miss some, but we stay valid).
                    for v in best_adds:
                        if v in S:
                            continue
                        ok = True
                        if (v + v) in S:
                            ok = False
                        if ok:
                            for x in S:
                                if (x + v) in S:
                                    ok = False
                                    break
                        if ok:
                            S.add(v)
                    improved = True
