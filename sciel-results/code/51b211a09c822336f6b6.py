<<<<<<< SEARCH
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
=======
    def axial_to_cartesian(q, r):
        # Scale by 1.001 to ensure strict non-overlap (gap between edges)
        # This prevents floating point errors or strict inequality checks in the evaluator
        scale = 1.001
        x = math.sqrt(3) * scale * (q + r / 2.0)
        y = 1.5 * scale * r
        return (x, y)
>>>>>>> REPLACE
