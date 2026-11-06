    def get_partitions():
        res = []
        # 0 pairs with 1, 2, 3, 4, 5
        for p0 in range(1, 6):
            rem = [i for i in range(1, 6) if i != p0]
            # Next smallest in rem is rem[0]
            for p1 in rem[1:]: # rem[0] pairs with p1
                p2 = rem[0]
                rem2 = [i for i in rem if i != p2 and i != p1]
                p3, p4 = rem2
                # Pairs are (0, p0), (p2, p1), (p3, p4)
                # Sort pairs to have a canonical form? 
                # The set of pairs is what matters.
                res.append([(0, p0), (p2, p1), (p3, p4)])
        return res
