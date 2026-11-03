<<<<<<< SEARCH
    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checks = 50
            for _ in range(n_checks):
                if time.time() > deadline - 5.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 5:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)
                r = candidates[0]

                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1

                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)
=======
    def two_for_one_search(blocks, seed, n_checks=80):
        """
        Try to replace two blocks with one.
        Improved: score candidates by coverage bonus.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for _ in range(n_checks):
                if time.time() > deadline - 5.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)

                # Score candidates by how many extra tsubs they cover
                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    # Bonus: tsubs covered by this block that are currently covered >= 2
                    bonus = 0
                    for ts_idx in block_tsubs[idx]:
                        if cover_count[ts_idx] >= 2:
                            bonus += 1
                    candidates.append((bonus, idx))
                    m ^= lsb
                    if len(candidates) > 20:
                        break

                if not candidates:
                    continue

                # Sort by bonus descending
                candidates.sort(key=lambda x: (-x[0], rng.random()))
                r = candidates[0][1]

                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1

                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)
>>>>>>> REPLACE

