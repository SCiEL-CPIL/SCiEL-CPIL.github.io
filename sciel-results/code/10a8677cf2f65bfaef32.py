            for i in range(len(other_indices)):
                q = other_indices[i]
                w = words[i]
                b = bit_masks[i]
                masks[q, w] &= ~b
