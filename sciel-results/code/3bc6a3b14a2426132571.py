<<<<<<< SEARCH
    def add(self, x):
        self.els |= (1 << x)
        self.sums |= (1 << (2 * x))
        self.count += 1
        temp = self.els & ~(1 << x)
        while temp:
            a = (temp & -temp).bit_length() - 1
            self.sums |= (1 << (a + x))
            self.diffs |= (1 << (x - a + 256))
            self.diffs |= (1 << (a - x + 256))
            temp &= temp - 1

    def remove(self, x):
        self.els &= ~(1 << x)
        self.sums &= ~(1 << (2 * x))
        self.count -= 1
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            self.sums &= ~(1 << (a + x))
            self.diffs &= ~(1 << (x - a + 256))
            self.diffs &= ~(1 << (a - x + 256))
            temp &= temp - 1

    def size(self):
        return self.count

    def get_els_set(self):
        s = set()
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            s.add(a)
            temp &= temp - 1
        return s
=======
    def add(self, x):
        self.els |= (1 << x)
        self.sums |= (1 << (2 * x))
        self.count += 1
        temp = self.els & ~(1 << x)
        while temp:
            a = (temp & -temp).bit_length() - 1
            self.sums |= (1 << (a + x))
            self.diffs |= (1 << (x - a + 256))
            self.diffs |= (1 << (a - x + 256))
            temp &= temp - 1

    def remove(self, x):
        self.els &= ~(1 << x)
        self.sums &= ~(1 << (2 * x))
        self.count -= 1
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            self.sums &= ~(1 << (a + x))
            self.diffs &= ~(1 << (x - a + 256))
            self.diffs &= ~(1 << (a - x + 256))
            temp &= temp - 1

    def size(self):
        return self.count

    def get_els_set(self):
        s = set()
        temp = self.els
        while temp:
            a = (temp & -temp).bit_length() - 1
            s.add(a)
            temp &= temp - 1
        return s

    def rebuild_from_set(self, s):
        """Rebuild entire state from a Python set. O(k^2) but safe."""
        self.els = 0
        self.sums = 0
        self.diffs = 0
        self.count = len(s)
        lst = sorted(s)
        for x in lst:
            self.els |= (1 << x)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                a, b = lst[i], lst[j]
                self.sums |= (1 << (a + b))
                if a != b:
                    self.diffs |= (1 << (a - b + 256))
                    self.diffs |= (1 << (b - a + 256))
>>>>>>> REPLACE

