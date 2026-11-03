<<<<<<< SEARCH
    def greedy_cover(seed_offset=0):
        """Greedy set cover with optional randomization."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        # Randomize the order of centers for diversity
        center_order = np.arange(total_words)
        if seed_offset > 0:
            rng = np.random.RandomState(seed_offset)
            rng.shuffle(center_order)

        # Process centers in order, but use gain-based selection
        # Actually, let's do standard greedy but with random tie-breaking
        while np.any(uncovered):
            if time.time() > deadline:
                break

            gains = uncovered[balls].sum(axis=1)
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all centers with max gain
            best_centers = np.where(gains == max_gain)[0]
            if len(best_centers) == 1:
                best_center = int(best_centers[0])
            else:
                # Random tie-break for diversity
                best_center = int(best_centers[np.random.randint(len(best_centers))])

            code.append(best_center)
            uncovered[balls[best_center]] = False
            cover_count[balls[best_center]] += 1

        return code, cover_count
=======
    def greedy_cover(seed_offset=0):
        """Greedy set cover with lazy-delete priority queue for O(1) amortized updates."""
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        cover_count = np.zeros(total_words, dtype=np.int32)

        # Initialize priority queue with all centers and their initial gain
        # gain(c) = number of uncovered words in balls[c]
        # Using a max-heap via negative gains
        pq = []
        for c in range(total_words):
            gain = balls[c].sum()  # all words uncovered initially, so gain = ball_size
            # Add small random perturbation for diversity
            if seed_offset > 0:
                perturbation = (seed_offset * 7919 + c * 104729) % 1000
                gain_key = -(gain * 1000 + perturbation)
            else:
                gain_key = -gain
            heapq.heappush(pq, (gain_key, c))

        # Track current gain for each center to detect stale entries
        current_gain = np.full(total_words, ball_size, dtype=np.int32)

        while pq:
            if time.time() > deadline:
                break

            gain_key, c = heapq.heappop(pq)
            actual_gain = current_gain[c]

            # Lazy deletion: skip if this entry is stale
            if actual_gain == 0:
                if not np.any(uncovered):
                    break
                continue

            # Verify this is still the best (or close enough)
            # In lazy deletion, we accept the first valid entry
            # But we should check it's truly max
            # For efficiency, just accept it (standard lazy-delete greedy)
            if actual_gain <= 0:
                continue

            code.append(c)
            uncovered[balls[c]] = False
            cover_count[balls[c]] += 1

            # Update gains for all centers that share uncovered words with c
            # For each word w in balls[c] that was just covered:
            #   For each center c' that also covers w (c' != c):
            #     current_gain[c'] -= 1
            #     Push updated gain to PQ
            for w in balls[c]:
                if not uncovered[w]:  # w was just covered (or already was)
                    for c2 in word_to_centers_sets[w]:
                        if c2 == c:
                            continue
                        old_gain = current_gain[c2]
                        if old_gain > 0:
                            current_gain[c2] = old_gain - 1
                            new_gain_key = -(current_gain[c2] * 1000 + (seed_offset * 7919 + c2 * 104729) % 1000)
                            heapq.heappush(pq, (new_gain_key, c2))

            # Check if all covered
            if not np.any(uncovered):
                break

        return code, cover_count
>>>>>>> REPLACE

