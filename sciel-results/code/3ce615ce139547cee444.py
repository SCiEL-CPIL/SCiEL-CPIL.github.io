    def greedy_init():
        uncovered = set(range(num_subsets))
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to anchor
            anchor_sid = random.choice(list(uncovered))
            # Consider all blocks covering this anchor
            candidates = blocks_by_subset[anchor_sid]
            best_block = None
            best_gain = 0
            for b in candidates:
                gain = 0
                for sid in block_to_subs[b]:
                    if sid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                    if best_gain == len(block_to_subs[best_block]):
                        break
            if best_block:
                blocks.append(best_block)
                for sid in block_to_subs[best_block]:
                    uncovered.discard(sid)
        return blocks
