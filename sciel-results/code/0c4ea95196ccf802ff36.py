    def is_bad_k4(quad, colors):
        q = tuple(sorted(quad))
        return (colors[triple_index[(q[0], q[1], q[2])]] == 0 and
                colors[triple_index[(q[0], q[1], q[3])]] == 0 and
                colors[triple_index[(q[0], q[2], q[3])]] == 0 and
                colors[triple_index[(q[1], q[2], q[3])]] == 0)
