class ValidSet:
    def __init__(self):
        self.elements = []
        self.triple_sums = [] # List of all sums of 3 distinct elements
        self.pair_sums_mod = {} # Dict: a -> set of (u+v) % a for u,v in elements \ {a}
        
    def is_valid_addition(self, x):
        n = len(self.elements)
        if n < 3:
            return True
            
        # 1. Check x: x should not divide any triple sum
        # Optimization: If x is very large, it might not divide any sum?
        # Max sum is 3 * 192 = 576.
        # If x > 576, it can't divide any sum (since sums are positive).
        # But x <= 192.
        # So we just iterate.
        for s in self.triple_sums:
            if s % x == 0:
                return False

        # 2. Check existing a: a should not divide x + u + v
        for a in self.elements:
            target = (-x) % a
            if target in self.pair_sums_mod[a]:
                return False
        return True

    def add(self, x):
        # Update state
        n = len(self.elements)
        
        # 1. Update triple_sums
        # New triples involving x: (x, u, v) where u, v in elements
        for i in range(n):
            for j in range(i+1, n):
                self.triple_sums.append(x + self.elements[i] + self.elements[j])
        
        # 2. Update pair_sums_mod
        # For each existing a, add new pairs (x, u) and (u, x) -> just (x+u)%a
        for a in self.elements:
            if a not in self.pair_sums_mod:
                self.pair_sums_mod[a] = set()
            for u in self.elements:
                if u != a:
                    self.pair_sums_mod[a].add((x + u) % a)
            
        # Initialize pair_sums_mod for x
        self.pair_sums_mod[x] = set()
        for i in range(n):
            for j in range(i+1, n):
                u = self.elements[i]
                v = self.elements[j]
                self.pair_sums_mod[x].add((u + v) % x)
                
        self.elements.append(x)
