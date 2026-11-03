        for idx in candidate_indices:
            # Calculate gain
            # block_triples[idx] is a list of tuples
            # uncovered is a set of tuples
            gain = len(uncovered.intersection(block_triples[idx]))
            
            if gain > best_gain:
                best_gain = gain
                best_blocks = [idx]
            elif gain == best_gain:
                best_blocks.append(idx)
