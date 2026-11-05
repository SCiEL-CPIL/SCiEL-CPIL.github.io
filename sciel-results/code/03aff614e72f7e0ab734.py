            for i, mask in enumerate(block_masks):
                gain = (mask & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
                    tie_count = 1
                elif gain == max_gain and gain > 0:
                    tie_count += 1
                    if randomize_ties and random.random() < 1.0 / tie_count:
                        best_block_idx = i
