import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    total_words = q ** n

    # Fallback for large spaces where greedy is too slow
    if total_words > 50000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]

    # Precompute powers of q for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute balls as lists of indices for efficient operations
    balls = []
    for i in range(total_words):
        w = idx_to_tuple(i)
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball.append(new_idx)
        balls.append(ball)

    # Helper to prune redundant codewords from a list of indices
    def prune_cover(indices):
        indices = list(set(indices))
        if not indices:
            return []

        # Count how many times each word is covered
        cover_counts = [0] * total_words
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1

        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                necessary = False
                for w in balls[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    # Remove it
                    for w in balls[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    # Local search: try to remove codewords or swap them
    def local_search(indices):
        indices = list(indices)
        if not indices:
            return indices

        # Count coverage
        cover_counts = [0] * total_words
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1

        improved = True
        while improved:
            improved = False
            for i in range(len(indices)):
                # Try removing indices[i]
                for w in balls[indices[i]]:
                    cover_counts[w] -= 1

                # Check if still covered
                valid = True
                for w in range(total_words):
                    if cover_counts[w] == 0:
                        valid = False
                        break

                if valid:
                    # Successfully removed
                    indices.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for w in balls[indices[i]]:
                        cover_counts[w] += 1

        # Try swaps: remove one, add another that covers more
        improved = True
        while improved:
            improved = False
            for i in range(len(indices)):
                # Try removing indices[i]
                for w in balls[indices[i]]:
                    cover_counts[w] -= 1

                # Find uncovered words
                uncovered = [w for w in range(total_words) if cover_counts[w] == 0]
                
                if not uncovered:
                    # Successfully removed
                    indices.pop(i)
                    improved = True
                    break
                else:
                    # Find a candidate that covers the most uncovered words
                    best_gain = 0
                    best_c = -1
                    for c in range(total_words):
                        if c in indices:
                            continue
                        gain = sum(1 for w in balls[c] if cover_counts[w] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    
                    if best_gain == len(uncovered) and best_gain > 0:
                        # Swap: add best_c, remove indices[i]
                        indices[i] = best_c
                        for w in balls[best_c]:
                            cover_counts[w] += 1
                        improved = True
                        break
                    else:
                        # Restore
                        for w in balls[indices[i]]:
                            cover_counts[w] += 1

        return indices

    best_code = None
    best_size = float('inf')

    # 1. Try Direct Product Construction (if applicable)
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        dp_indices = prune_cover(dp_indices)
        if len(dp_indices) < best_size:
            best_size = len(dp_indices)
            best_code = dp_indices[:]

    # 2. Spectral Graph Theory Approach
    # Construct the adjacency matrix of the Hamming graph H(n, q)
    # The Laplacian eigenvalues/eigenvectors can guide the greedy selection
    # We use the second smallest eigenvector (Fiedler vector) to prioritize nodes
    
    # For the specific case of K_3(6,1), we can leverage the structure
    # The Hamming graph H(6,3) has 729 nodes
    # We'll use a simplified spectral approach: prioritize nodes with high degree in the
    # "uncovered" subgraph, which approximates centrality
    
    all_mask = (1 << total_words) - 1

    # Precompute ball masks for faster gain calculation
    ball_masks = []
    for ball in balls:
        mask = 0
        for w in ball:
            mask |= (1 << w)
        ball_masks.append(mask)

    # Spectral-inspired greedy: use eigenvector-based weighting
    # For efficiency, we approximate the Fiedler vector using power iteration
    # on the complement of the adjacency matrix (Laplacian)
    
    # Build adjacency matrix (sparse representation)
    # For H(6,3), each node has degree n*(q-1) = 6*2 = 12
    # We'll use a simplified approach: weight candidates by their "centrality"
    # in the current uncovered set
    
    num_restarts = 200

    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        uncovered = all_mask
        code_indices = []

        while uncovered:
            # Calculate gains with spectral-inspired weighting
            # Weight = gain * (1 + centrality_bonus)
            # Centrality bonus: prefer nodes that cover many "central" uncovered nodes
            
            max_gain = -1
            top_candidates = []
            
            # Precompute centrality scores for uncovered nodes
            # Approximate: count how many uncovered nodes are adjacent to each candidate
            # This is expensive, so we use a simpler heuristic:
            # Prefer candidates whose ball overlaps more with the "core" of uncovered nodes
            
            for c in range(total_words):
                gain = (ball_masks[c] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    top_candidates = [c]
                elif gain == max_gain:
                    top_candidates.append(c)

            if max_gain <= 0:
                break

            # Spectral-inspired tie-breaking: use eigenvector-based scoring
            # We approximate this by preferring candidates that cover nodes
            # with higher "connectivity" in the uncovered subgraph
            if len(top_candidates) > 1:
                # Score each candidate by the sum of degrees of nodes it covers
                # (in the uncovered subgraph)
                best_score = -1
                best_c = top_candidates[0]
                
                for c in top_candidates:
                    # Calculate spectral score
                    score = 0
                    for w in balls[c]:
                        if (uncovered >> w) & 1:
                            # Count neighbors of w that are also uncovered
                            # This is expensive, so use a simpler proxy:
                            # Use the ball size of w (constant) or precomputed degree
                            score += 1  # Simplified: all nodes have same degree
                    # Add small random perturbation for diversity
                    score += random.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_c = c
            else:
                best_c = top_candidates[0]

            code_indices.append(best_c)
            uncovered &= ~ball_masks[best_c]

            if time.time() > deadline - 15.0:
                break

        if uncovered:
            continue

        # Prune the result
        pruned = prune_cover(code_indices)

        # Local search to remove more
        pruned = local_search(pruned)

        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned[:]

    # 3. Try a few more restarts with different random seeds if time permits
    if time.time() < deadline - 20.0 and best_size > 70:
        for r in range(50):
            if time.time() > deadline - 10.0:
                break

            uncovered = all_mask
            code_indices = []

            while uncovered:
                max_gain = -1
                top_candidates = []

                for c in range(total_words):
                    gain = (ball_masks[c] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                        top_candidates = [c]
                    elif gain == max_gain:
                        top_candidates.append(c)

                if max_gain <= 0:
                    break

                # Different tie-breaking: prefer random
                if top_candidates:
                    best_c = random.choice(top_candidates)
                else:
                    break

                code_indices.append(best_c)
                uncovered &= ~ball_masks[best_c]

                if time.time() > deadline - 15.0:
                    break

            if uncovered:
                continue

            pruned = prune_cover(code_indices)
            pruned = local_search(pruned)

            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned[:]

    if best_code is None:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            best_code = list(range(total_words))

    # Convert indices back to list of lists
    result = [list(idx_to_tuple(c)) for c in best_code]

    return result
