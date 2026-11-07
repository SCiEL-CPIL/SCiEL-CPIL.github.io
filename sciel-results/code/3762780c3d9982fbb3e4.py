    def score_candidate(x, current_sums):
        # Prefer x not dividing any sum and not being divided by any sum
        # And prefer x such that s+x doesn't create divisibility
        # For simplicity, just prefer larger x?
        return -x
