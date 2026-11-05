            best_gain = 0
            candidates = []
            for b in range(num_blocks):
                gain = 0
                for idx in block_covered[b]:
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [b]
                elif gain == best_gain and gain > 0:
                    candidates.append(b)
            
            if not candidates:
                break
            best_block = random.choice(candidates)
