            gains = np.zeros(len(candidates), dtype=np.uint64)
            for w in range(num_words):
                gains += np.bitwise_and(cand_masks[:, w], uncovered[w])
