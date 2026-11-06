    def can_add(S, x):
        Slist = sorted(S)
        # Triples where x is smallest: x < b <= c
        for i, b in enumerate(Slist):
            if b <= x: continue
            for j in range(i, len(Slist)):
                c = Slist[j]
                if c < b: continue
                if (b + c) % x == 0: return False
        # Triples where x is middle: a < x <= c
        for a in Slist:
            if a >= x: continue
            for c in Slist:
                if c < x: continue
                if (x + c) % a == 0: return False
        # Triples where x is largest: a < b <= x
        for i, a in enumerate(Slist):
            if a >= x: continue
            for j in range(i+1, len(Slist)):
                b = Slist[j]
                if b > x: break
                if (b + x) % a == 0: return False
        return True
