    def construct():
        # Assign values 1..16 to vertices 0..15
        # Color (i, j) is (val[i] * val[j]) % 17
        vals = [i + 1 for i in range(16)]
        colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                colors.append((vals[i] * vals[j]) % 17)
        return colors
    