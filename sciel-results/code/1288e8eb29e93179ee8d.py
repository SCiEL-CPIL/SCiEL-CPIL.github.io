"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Greedy minimal extension: iteratively pick the smallest positive integer
    not yet used that does not cause any subset-sum collision.
    """
    deadline = time.time() + 80  # safety margin before 90s kill

    # Start with the smallest possible: 1
    elements = [1]
    # All subset sums of current elements (start with empty subset sum = 0)
    subset_sums = {0, 1}

    for _ in range(11):  # need 12 total, already have 1
        candidate = 2  # smallest unused positive integer (1 is taken)
        while True:
            # Check if adding 'candidate' creates any collision
            # New subset sums = old subset_sums ∪ {s + candidate for s in subset_sums}
            # Collision exists if any (s + candidate) is already in subset_sums
            # i.e., if (candidate - s1) == s2 for some s1, s2 in subset_sums
            # More directly: check if any (s + candidate) is in subset_sums
            has_collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    has_collision = True
                    break

            if not has_collision:
                break

            candidate += 1
            # Deadline check
            if time.time() > deadline:
                # Return current best (may be incomplete, but task says minimize largest)
                # If we haven't found 12, fall back to powers of 2
                return [1 << i for i in range(12)]

        # Add candidate
        elements.append(candidate)
        # Update subset sums
        new_sums = set()
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums = subset_sums | new_sums

    return elements
# EVOLVE-BLOCK-END
