                # If time ran out or loop broke early, we have a partial code.
                # We must ensure validity.
                # Add codewords for all uncovered words to guarantee coverage.
                for i in range(total_words):
                    if not covered[i]:
                        code.append(words[i])
                        # Update coverage for neighbors of the added word
                        for j in neighbors[i]:
                            covered[j] = 1
