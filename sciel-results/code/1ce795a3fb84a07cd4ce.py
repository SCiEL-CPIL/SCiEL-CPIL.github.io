<<<<<<< SEARCH
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
=======
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        # The girth of CD(d, q) is 2d + 1.
        # We require girth >= g, so 2d + 1 >= g => d >= (g - 1) / 2.
        # We choose the smallest integer d satisfying this, to minimize n = 2 * q^d.
        d = (g - 1 + 1) // 2
        if d < 3:
            d = 3
>>>>>>> REPLACE
