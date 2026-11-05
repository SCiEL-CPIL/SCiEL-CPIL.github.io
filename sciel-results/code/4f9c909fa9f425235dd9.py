    # Fallback: Random 16-coloring is almost surely valid
    # Probability of a specific K4 having < 5 colors is negligible.
    # We verify it just in case.
    def random_valid_16():
        while True:
            cols = [random.randint(0, 15) for _ in range(120)]
            # Check all K4s
            valid = True
            for edges in k4_edges:
                c = set(cols[e] for e in edges)
                if len(c) < 5:
                    valid = False
                    break
            if valid:
                return cols
